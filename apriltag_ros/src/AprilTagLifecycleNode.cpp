// MonKey fork: lifecycle variant of AprilTagNode. onCamera, onParameter and
// getTagInfoFromDatabase are copies of AprilTagNode's (src/AprilTagNode.cpp), and on_configure
// declares the same parameters in the same order: keep them in sync when merging upstream.

#include "AprilTagLifecycleNode.hpp"

#include <cmath>
#include <optional>

// ros
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <image_transport/camera_common.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sqlite3.h>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// apriltag
#include "tag_functions.hpp"


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

namespace {

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}

rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

const std::unordered_map<std::string, rmw_qos_profile_t> qos_profiles{
    {"default", rmw_qos_profile_default},
    {"sensor_data", rmw_qos_profile_sensor_data},
    {"system_default", rmw_qos_profile_system_default},
};

} // namespace

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagLifecycleNode)


AprilTagLifecycleNode::AprilTagLifecycleNode(const rclcpp::NodeOptions& options)
  : LifecycleNode("apriltag", options)
{}

AprilTagLifecycleNode::~AprilTagLifecycleNode()
{
    releaseActive();
    releaseConfigured();
}

AprilTagLifecycleNode::CallbackReturn
AprilTagLifecycleNode::on_configure(const rclcpp_lifecycle::State&)
{
    try {
        configureDetector();
    }
    catch(const std::exception& e) {
        RCLCPP_ERROR_STREAM(get_logger(), "Configuration failed: " << e.what());
        releaseConfigured();
        return CallbackReturn::FAILURE;
    }
    return CallbackReturn::SUCCESS;
}

AprilTagLifecycleNode::CallbackReturn
AprilTagLifecycleNode::on_activate(const rclcpp_lifecycle::State&)
{
    try {
        // TF Buffer
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        pub_detections->on_activate();
        pub_pose_array->on_activate();
        startCamera();
    }
    catch(const std::exception& e) {
        RCLCPP_ERROR_STREAM(get_logger(), "Activation failed: " << e.what());
        releaseActive();
        return CallbackReturn::FAILURE;
    }
    return CallbackReturn::SUCCESS;
}

AprilTagLifecycleNode::CallbackReturn
AprilTagLifecycleNode::on_deactivate(const rclcpp_lifecycle::State&)
{
    releaseActive();
    return CallbackReturn::SUCCESS;
}

AprilTagLifecycleNode::CallbackReturn
AprilTagLifecycleNode::on_cleanup(const rclcpp_lifecycle::State&)
{
    releaseConfigured();
    return CallbackReturn::SUCCESS;
}

AprilTagLifecycleNode::CallbackReturn
AprilTagLifecycleNode::on_shutdown(const rclcpp_lifecycle::State&)
{
    releaseActive();
    releaseConfigured();
    return CallbackReturn::SUCCESS;
}

// Parameters stay declared after on_cleanup (read-only ones cannot be undeclared). A second
// on_configure reads the current value and passes it through onParameter, as declare_parameter
// does through the registered callback on the first one.
template<typename T>
T AprilTagLifecycleNode::declare(const std::string& name, const T& default_value, const rcl_interfaces::msg::ParameterDescriptor& descriptor)
{
    if(!has_parameter(name)) {
        return declare_parameter(name, default_value, descriptor);
    }
    const rclcpp::Parameter parameter = get_parameter(name);
    onParameter({parameter});
    return parameter.get_value<T>();
}

// AprilTagNode's constructor, without the camera subscription and the TF listener
void AprilTagLifecycleNode::configureDetector()
{
    td = apriltag_detector_create();
    // parameter
    cb_parameter = add_on_set_parameters_callback(std::bind(&AprilTagLifecycleNode::onParameter, this, std::placeholders::_1));

    // camera subscription, created on activation
    image_transport_ = declare<std::string>("image_transport", "raw", descr({}, true));
    qos_profile_ = declare<std::string>("qos_profile", "default", descr("qos profile to use. 'default', 'sensor_data' or 'system_default'", true));
    if(!qos_profiles.count(qos_profile_)) {
        throw std::runtime_error("Unknown qos profile '" + qos_profile_ + "'");
    }

    // topics
    pub_detections = create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1));
    pub_pose_array = create_publisher<geometry_msgs::msg::PoseArray>("pose_array", rclcpp::QoS(1));
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(
#ifdef tf2_ros_NODE_INTERFACE
        tf2_ros::TransformBroadcaster::RequiredInterfaces{*this}
#else
        this
#endif
    );

    // read-only parameters
    const std::string tag_family = declare<std::string>("family", "36h11", descr("tag family", true));
    tag_edge_size = declare("size", 1.0, descr("default tag size", true));

    // get tag names, IDs and sizes (ignored for frames and sizes when use_database is true)
    const auto ids = declare("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto frames = declare("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
    const auto sizes = declare("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));
    tag_ids.insert(ids.begin(), ids.end());

    const std::vector<std::string> bundle_names = declare("tag_bundles.bundle_names", std::vector<std::string>{}, descr("tag bundle names", true));
    // get method for estimating tag pose
    const std::string pose_estimation_method =
        declare<std::string>("pose_estimation_method", "pnp",
                             descr("pose estimation method: \"pnp\" (more accurate) or \"homography\" (faster), "
                                   "set to \"\" (empty) to disable pose estimation",
                                   true));

    if(!pose_estimation_method.empty()) {
        if(pose_estimation_methods.count(pose_estimation_method)) {
            estimate_pose = pose_estimation_methods.at(pose_estimation_method);
        }
        else {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown pose estimation method '" << pose_estimation_method << "'.");
        }
    }

    // detector parameters in "detector" namespace
    declare("detector.threads", td->nthreads, descr("number of threads"));
    declare("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare("detector.debug", td->debug, descr("write additional debugging images to working directory"));

    declare("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare("profile", false, descr("print profiling information to stdout"));

    map_frame_ = declare<std::string>("map_frame", "map", descr("Fixed frame of the pose_array output", true));
    publish_tf_ = declare("publish_tf", true, descr("Whether to publish tag frame tf", true));
    // Prepended to every published TF child frame (tags and bundles), e.g. "<robot_id>/",
    // so detectors on several robots do not publish the same frame names on the global /tf.
    frame_prefix_ = declare<std::string>("frame_prefix", "", descr("prefix of published tag and bundle frames", true));

    for(const std::string& bundle_name : bundle_names) {
        TagBundlePtr bundle = std::make_shared<TagBundle>();

        bundle->frame_id = frame_prefix_ + bundle_name;
        const std::vector<int64_t> bundle_ids_vector = declare("tag_bundles." + bundle_name + ".ids", std::vector<int64_t>{}, descr("bundle ids", true));
        bundle->ids = std::set<int64_t>(bundle_ids_vector.begin(), bundle_ids_vector.end());
        for(const int64_t& id : bundle->ids) {
            tag_id_to_bundles[id].push_back(bundle);
            const std::string prefix = "tag_bundles." + bundle_name + "." + std::to_string(id);
            bundle->id_to_size[id] = declare(prefix + ".size", 1.0, descr("bundle size", true));
            bundle->id_to_tf[id] = declare(prefix + ".transform", std::vector<double>{}, descr("bundle transform", true));
        }
        all_tag_bundles.push_back(bundle);
    }

    const auto use_database = declare("use_database", false, descr("Whether to get tag info from SQLite database", true));
    if (use_database) {
        // Get tag frames and sizes from SQLite database
        getTagInfoFromDatabase(tag_frames, tag_sizes);
    } else {
        if(!frames.empty()) {
            if(ids.size() != frames.size()) {
                throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
            }
            for(size_t i = 0; i < ids.size(); i++) { tag_frames[ids[i]] = frames[i]; }
        }

        if(!sizes.empty()) {
            // use tag specific size
            if(ids.size() != sizes.size()) {
                throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
            }
            for(size_t i = 0; i < ids.size(); i++) { tag_sizes[ids[i]] = sizes[i]; }
        }
    }

    if(tag_fun.count(tag_family)) {
        tf = tag_fun.at(tag_family).first();
        tf_destructor = tag_fun.at(tag_family).second;
        apriltag_detector_add_family(td, tf);
    }
    else {
        throw std::runtime_error("Unsupported tag family: " + tag_family);
    }
}

void AprilTagLifecycleNode::startCamera()
{
    // AprilTagNode subscribes on itself, so its remapping rules apply to every topic its
    // CameraSubscriber derives from the image topic. The helper node takes no command-line
    // arguments; it gets one rule for each derived topic (the transport topic and the sibling
    // camera_info), mapping it to the name this node resolves it to, which lands both
    // subscriptions where AprilTagNode's would.
    const std::string image_topic = this->get_node_topics_interface()->resolve_topic_name("image_rect");
    const std::vector<std::string> derived_topics{
        image_transport_ == "raw" ? image_topic : image_topic + "/" + image_transport_,
        image_transport::getCameraInfoTopic(image_topic),
    };
    std::vector<std::string> arguments;
    for(const std::string& topic : derived_topics) {
        const std::string resolved = this->get_node_topics_interface()->resolve_topic_name(topic);
        if(resolved != topic) {
            if(arguments.empty()) { arguments.push_back("--ros-args"); }
            arguments.insert(arguments.end(), {"-r", topic + ":=" + resolved});
        }
    }
    // this node's parameter overrides, for parameters the transport plugin declares
    std::vector<rclcpp::Parameter> overrides;
    for(const auto& parameter_override : this->get_node_parameters_interface()->get_parameter_overrides()) {
        overrides.emplace_back(parameter_override.first, parameter_override.second);
    }

    const rclcpp::Context::SharedPtr context = this->get_node_base_interface()->get_context();
    rclcpp::NodeOptions options;
    options.context(context)
        .use_global_arguments(false)
        .arguments(arguments)
        .parameter_overrides(overrides)
        .start_parameter_services(false)
        .start_parameter_event_publisher(false)
        .use_intra_process_comms(this->get_node_options().use_intra_process_comms());
    camera_node_ = std::make_shared<rclcpp::Node>(std::string(this->get_name()) + "_camera", this->get_namespace(), options);

    sub_cam = image_transport::CameraSubscriber(
#ifdef image_transport_NODE_INTERFACE
        image_transport::RequiredInterfaces{*camera_node_},
#else
        camera_node_.get(),
#endif
        image_topic,
        std::bind(&AprilTagLifecycleNode::onCamera, this, std::placeholders::_1, std::placeholders::_2),
        image_transport_,
#ifdef image_transport_QoS
        rclcpp::QoS{rclcpp::QoSInitialization::from_rmw(
#endif
            qos_profiles.at(qos_profile_)
#ifdef image_transport_QoS
                )}
#endif
    );

    rclcpp::ExecutorOptions executor_options;
    executor_options.context = context;
    camera_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(executor_options);
    camera_executor_->add_node(camera_node_);
    camera_running_ = true;
    camera_thread_ = std::thread([this, executor = camera_executor_, context]() {
        while(camera_running_ && rclcpp::ok(context)) {
            executor->spin_once(std::chrono::milliseconds(100));
        }
    });
}

void AprilTagLifecycleNode::stopCamera()
{
    // stop callbacks first: onCamera uses the TF buffer and the publishers
    camera_running_ = false;
    if(camera_executor_) { camera_executor_->cancel(); }
    if(camera_thread_.joinable()) { camera_thread_.join(); }
    if(sub_cam) { sub_cam.shutdown(); }
    sub_cam = image_transport::CameraSubscriber();
    camera_executor_.reset();
    camera_node_.reset();
}

void AprilTagLifecycleNode::releaseActive()
{
    stopCamera();
    if(pub_detections) { pub_detections->on_deactivate(); }
    if(pub_pose_array) { pub_pose_array->on_deactivate(); }
    tf_listener_.reset();
    tf_buffer_.reset();
}

void AprilTagLifecycleNode::releaseConfigured()
{
    if(cb_parameter) {
        remove_on_set_parameters_callback(cb_parameter.get());
        cb_parameter.reset();
    }
    tf_broadcaster.reset();
    pub_pose_array.reset();
    pub_detections.reset();
    if(td) {
        apriltag_detector_destroy(td);
        td = nullptr;
    }
    if(tf) {
        tf_destructor(tf);
        tf = nullptr;
    }
    estimate_pose = nullptr;
    tag_ids.clear();
    all_tag_bundles.clear();
    tag_id_to_bundles.clear();
    tag_frames.clear();
    tag_sizes.clear();
}

void AprilTagLifecycleNode::onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img,
                                     const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci)
{
    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {msg_ci->p[0], msg_ci->p[5], msg_ci->p[2], msg_ci->p[6]};

    // check for valid intrinsics
    const bool calibrated = msg_ci->width && msg_ci->height &&
                            intrinsics[0] && intrinsics[1] && intrinsics[2] && intrinsics[3];

    if(estimate_pose != nullptr && !calibrated) {
        RCLCPP_WARN_STREAM(get_logger(), "The camera is not calibrated! Set 'pose_estimation_method' to \"\" (empty) to disable pose estimation and this warning.");
    }

    // convert to 8bit monochrome image
    const cv::Mat img_uint8 = cv_bridge::toCvShare(msg_img, "mono8")->image;

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    std::unordered_map<TagBundlePtr, std::vector<apriltag_detection_t*>> bundle_detections;

    // Tag positions in map_frame_. Published for every processed image, empty when no tag passed.
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp = msg_img->header.stamp;
    pose_array.header.frame_id = map_frame_;
    // camera -> map_frame_ at the image time, looked up once per image when a tag needs it
    std::optional<geometry_msgs::msg::TransformStamped> camera_to_map;
    bool camera_to_map_failed = false;

    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

        if(tag_id_to_bundles.count(det->id)) {
            for(const TagBundlePtr& bundle : tag_id_to_bundles[det->id]) {
                bundle_detections[bundle].push_back(det);
            }
        }

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        // 3D orientation and position
        if(estimate_pose != nullptr && calibrated) {
            if(tag_id_to_bundles.count(det->id) && !tag_ids.count(det->id)) { continue; }
            geometry_msgs::msg::TransformStamped tf;
            tf.header = msg_img->header;
            // set child frame name by generic tag name or configured tag name
            tf.child_frame_id = frame_prefix_ + (tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id));
            const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
            tf.transform = estimate_pose(det, intrinsics, size);

            tfs.push_back(tf);

            if(!camera_to_map && !camera_to_map_failed) {
                try {
                    camera_to_map = tf_buffer_->lookupTransform(map_frame_, msg_img->header.frame_id,
                                                                rclcpp::Time(msg_img->header.stamp),
                                                                rclcpp::Duration::from_seconds(0.05));
                }
                catch(const tf2::TransformException& ex) {
                    camera_to_map_failed = true;
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No transform %s -> %s, tag poses not published: %s",
                                         msg_img->header.frame_id.c_str(), map_frame_.c_str(), ex.what());
                }
            }
            // Never publish a pose that was not transformed (it would sit at the map_frame_ origin).
            if(!camera_to_map) { continue; }

            geometry_msgs::msg::Pose pose_camera;
            pose_camera.position.x = tf.transform.translation.x;
            pose_camera.position.y = tf.transform.translation.y;
            pose_camera.position.z = tf.transform.translation.z;
            pose_camera.orientation.w = 1.0;
            geometry_msgs::msg::Pose pose_map;
            tf2::doTransform(pose_camera, pose_map, *camera_to_map);

            // Orientation: yaw only, the horizontal direction the printed face points.
            // A visible face always points back at the camera (negative optical z), which
            // fixes the sign whatever the tag axis convention and the in-plane tag rotation.
            tf2::Quaternion q_camera_tag;
            tf2::fromMsg(tf.transform.rotation, q_camera_tag);
            tf2::Vector3 normal = tf2::quatRotate(q_camera_tag, tf2::Vector3(0, 0, 1));
            if(normal.z() > 0) { normal = -normal; }
            tf2::Quaternion q_camera_to_map;
            tf2::fromMsg(camera_to_map->transform.rotation, q_camera_to_map);
            normal = tf2::quatRotate(q_camera_to_map, normal);
            tf2::Quaternion q_yaw;
            q_yaw.setRPY(0.0, 0.0, std::atan2(normal.y(), normal.x()));
            pose_map.orientation = tf2::toMsg(q_yaw);

            pose_array.poses.push_back(pose_map);

            // Pose with ID
            // apriltag_msgs::msg::AprilTagPoseId msg_pose_id;
            // msg_pose_id.header = msg_img->header;
            // msg_pose_id.family = std::string(det->family->name);
            // msg_pose_id.id = det->id;
            // msg_pose_id.pose = pose
            // pub_pose_id->publish(msg_pose_id);  // Publish pose with ID
        }
    }

    pub_detections->publish(msg_detections);
    pub_pose_array->publish(pose_array);

    if(estimate_pose != nullptr && calibrated && publish_tf_) {
        for(const auto& bundle_detection : bundle_detections) {
            const TagBundlePtr& bundle = bundle_detection.first;
            const std::vector<apriltag_detection_t*>& detections = bundle_detection.second;
            geometry_msgs::msg::TransformStamped bundle_transform_stamped;
            bundle_transform_stamped.header = msg_img->header;
            bundle_transform_stamped.child_frame_id = bundle->frame_id;
            bundle_transform_stamped.transform = pnp_bundle(detections, intrinsics, bundle->id_to_size, bundle->id_to_tf);
            tf_broadcaster->sendTransform(bundle_transform_stamped);
        }
        tf_broadcaster->sendTransform(tfs);
    }

    apriltag_detections_destroy(detections);
}

rcl_interfaces::msg::SetParametersResult
AprilTagLifecycleNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}

// SQLite query to get tag information (ID and Frame Name)
void AprilTagLifecycleNode::getTagInfoFromDatabase(
    std::unordered_map<int, std::string>& tag_frames, std::unordered_map<int, double>& tag_sizes) 
{
    sqlite3* db;
    sqlite3_stmt* stmt;

    // Get SQLite database path
    const auto path = declare<std::string>("database_path", "/root/database/apriltag.db3", descr("File path for SQLite database"));
    RCLCPP_INFO(get_logger(), "Reading tag database file %s", path.c_str());

    // Open the SQLite database
    int rc = sqlite3_open(path.c_str(), &db); // Specify your database path
    if (rc) {
        RCLCPP_ERROR(rclcpp::get_logger("AprilTagNode"), "Can't open database: %s", sqlite3_errmsg(db));
        return;
    }

    // Prepare SQL query to get both tag ID and corresponding frame name
    std::string sql = "SELECT id, edge_size, frame FROM Data;"; // Assuming the database table has these columns
    rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        RCLCPP_ERROR(rclcpp::get_logger("AprilTagNode"), "Failed to prepare statement: %s", sqlite3_errmsg(db));
        sqlite3_close(db);
        return;
    }

    // Iterate over the rows in the result
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int tag_id = sqlite3_column_int(stmt, 0);
        double edge_size = sqlite3_column_double(stmt, 1);
        const char* frame_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        // Store the tag ID, frame name, and size in the unordered maps
        tag_frames[tag_id] = std::string(frame_name);
        tag_sizes[tag_id] = edge_size;
    }

    // Clean up
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}