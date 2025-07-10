// ros
#include "pose_estimation.hpp"
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <apriltag_msgs/msg/april_tag_pose_id.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/camera_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp> 
#include <geometry_msgs/msg/pose_array.hpp> 
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <sqlite3.h> //SQLite Library

// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

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

class AprilTagNode : public rclcpp::Node {
public:
    AprilTagNode(const rclcpp::NodeOptions& options);

    ~AprilTagNode() override;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    apriltag_family_t* tf;
    apriltag_detector_t* const td;

    // parameter
    std::mutex mutex;
    double tag_edge_size;
    std::atomic<int> max_hamming;
    std::atomic<bool> profile;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;
    std::string map_frame_;
    bool publish_tf_;

    std::function<void(apriltag_family_t*)> tf_destructor;

    const image_transport::CameraSubscriber sub_cam;
    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    const rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_pose_array; // Updated this line
    // const rclcpp::Publisher<apriltag_msgs::msg::AprilTagPoseId>::SharedPtr pub_pose_id;
    tf2_ros::TransformBroadcaster tf_broadcaster;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    pose_estimation_f estimate_pose = nullptr;

    void onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);

    void getTagInfoFromDatabase(std::unordered_map<int, std::string>& tag_frames, std::unordered_map<int, double>& tag_sizes);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)


AprilTagNode::AprilTagNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTagNode::onParameter, this, std::placeholders::_1))),
    td(apriltag_detector_create()),
    // topics
    sub_cam(image_transport::create_camera_subscription(
        this,
        this->get_node_topics_interface()->resolve_topic_name("image_rect"),
        std::bind(&AprilTagNode::onCamera, this, std::placeholders::_1, std::placeholders::_2),
        declare_parameter("image_transport", "raw", descr({}, true)),
        rmw_qos_profile_sensor_data)),
    pub_detections(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    pub_pose_array(create_publisher<geometry_msgs::msg::PoseArray>("pose_array", rclcpp::QoS(1))), // Updated this line
    // pub_pose_id(create_publisher<apriltag_msgs::msg::AprilTagPoseId>("tag_pose_id", rclcpp::QoS(1))),
    tf_broadcaster(this)
{
    // read-only parameters
    const std::string tag_family = declare_parameter("family", "36h11", descr("tag family", true));
    tag_edge_size = declare_parameter("size", 1.0, descr("default tag size", true));

    // get method for estimating tag pose
    const std::string& pose_estimation_method =
        declare_parameter("pose_estimation_method", "pnp",
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

    map_frame_ = declare_parameter("map_frame", "map", descr("Name of fixed frame"));
    publish_tf_ = declare_parameter("publish_tf", true, descr("Whether to publish tag frame tf"));

    // detector parameters in "detector" namespace
    declare_parameter("detector.threads", td->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", td->debug, descr("write additional debugging images to working directory"));

    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));

    const auto use_database = declare_parameter("use_database", false, descr("Whether to get tag info from SQLite database"));
    if (use_database) {
        // Get tag frames and sizes from SQLite database
        getTagInfoFromDatabase(tag_frames, tag_sizes);  // Fetch both tag ID, frame, and size
    } else {
        // get tag names, IDs and sizes
        const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
        const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
        const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));

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

    // TF Buffer
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

AprilTagNode::~AprilTagNode()
{
    apriltag_detector_destroy(td);
    tf_destructor(tf);
}

void AprilTagNode::onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img,
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

    // No detections found
    if (zarray_size(detections) == 0) return;

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    
    // Initialize PoseArray message
    geometry_msgs::msg::PoseArray pose_array;
    // pose_array.header = msg_img->header;  // Set header for PoseArray
    pose_array.header.stamp = msg_img->header.stamp;  // Set header for PoseArray
    pose_array.header.frame_id = map_frame_;

    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

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
            // Compute TF
            geometry_msgs::msg::TransformStamped tf;
            tf.header = msg_img->header;
            // set child frame name by generic tag name or configured tag name
            tf.child_frame_id = tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id);
            const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
            tf.transform = estimate_pose(det, intrinsics, size);

            try {
                // Step 1: Get the transform from map to camera frame
                geometry_msgs::msg::TransformStamped tf_map_to_camera =
                    tf_buffer_->lookupTransform(msg_img->header.frame_id, "map", msg_img->header.stamp);

                // Step 2: Extract yaw angle (Z-axis rotation) from that transform
                tf2::Quaternion q_map_to_camera;
                tf2::fromMsg(tf_map_to_camera.transform.rotation, q_map_to_camera);

                double roll, pitch, yaw;
                tf2::Matrix3x3(q_map_to_camera).getRPY(roll, pitch, yaw);  // we only need yaw

                // Step 3: Apply the yaw angle as a rotation to the tag pose
                tf2::Quaternion q_tag_cam;
                tf2::fromMsg(tf.transform.rotation, q_tag_cam);

                tf2::Quaternion rotation;
                rotation.setRPY(roll, pitch, yaw);

                tf2::Quaternion q_tag_rotated = q_tag_cam * rotation;
                q_tag_rotated.normalize();

                // Step 4: Assign the rotated quaternion
                tf.transform.rotation = tf2::toMsg(q_tag_rotated);
            }
            catch (tf2::TransformException &ex) {
                RCLCPP_WARN(get_logger(), "Transform error: %s", ex.what());
            }

            // Add pose to PoseArray
            geometry_msgs::msg::PoseStamped pose;
            pose.header = msg_img->header;
            pose.pose.position.x = tf.transform.translation.x;
            pose.pose.position.y = tf.transform.translation.y;
            pose.pose.position.z = tf.transform.translation.z;
            pose.pose.orientation = tf.transform.rotation;

            geometry_msgs::msg::PoseStamped pose_map;
            pose_map.header.stamp = msg_img->header.stamp;
            pose_map.header.frame_id = map_frame_;
            try {
                tf_buffer_->transform(pose, pose_map, map_frame_);
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(get_logger(), "Could not transform docking pose to base_link: %s", ex.what());
            }

            tfs.push_back(tf); // Add tf to tf array
            pose_array.poses.push_back(pose_map.pose); // Add pose to the PoseArray

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
    pub_pose_array->publish(pose_array); // Publish the PoseArray message
    if (publish_tf_) tf_broadcaster.sendTransform(tfs);

    apriltag_detections_destroy(detections);
}

rcl_interfaces::msg::SetParametersResult
AprilTagNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
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
void AprilTagNode::getTagInfoFromDatabase(
    std::unordered_map<int, std::string>& tag_frames, std::unordered_map<int, double>& tag_sizes) 
{
    sqlite3* db;
    sqlite3_stmt* stmt;

    // Get SQLite database path
    const auto path = declare_parameter("database_path", "/root/database/apriltag.db3", descr("File path for SQLite database"));
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