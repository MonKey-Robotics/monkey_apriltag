#pragma once

// MonKey fork: lifecycle variant of AprilTagNode (src/AprilTagNode.cpp).
// While ACTIVE it has the same parameters, topics and TF output. The camera subscription and the
// TF listener only exist while ACTIVE, so an INACTIVE detector costs nothing.

#include "pose_estimation.hpp"
#include <apriltag.h>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <atomic>
#include <functional>
#include <geometry_msgs/msg/pose_array.hpp>
#include <image_transport/camera_subscriber.hpp>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <set>
#include <string>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <thread>
#include <unordered_map>
#include <vector>


class AprilTagLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
    explicit AprilTagLifecycleNode(const rclcpp::NodeOptions& options);

    ~AprilTagLifecycleNode() override;

    // declare the parameters and build the detector, no camera subscription
    CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
    // create the TF listener and the camera subscription, activate the publishers
    CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
    // destroy the camera subscription and the TF listener, deactivate the publishers
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    // release the detector and the publishers (parameters stay declared)
    CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
    struct TagBundle
    {
        std::set<int64_t> ids;
        std::unordered_map<int, double> id_to_size;
        std::unordered_map<int, std::vector<double>> id_to_tf;
        std::string frame_id;
    };

    typedef std::shared_ptr<TagBundle> TagBundlePtr;
    typedef std::vector<TagBundlePtr> TagBundleVec;

    OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    apriltag_family_t* tf = nullptr;
    apriltag_detector_t* td = nullptr;

    // parameter
    std::mutex mutex;
    double tag_edge_size = 1.0;
    std::atomic<int> max_hamming{0};
    std::atomic<bool> profile{false};
    std::set<int64_t> tag_ids;
    TagBundleVec all_tag_bundles;
    std::unordered_map<int64_t, TagBundleVec> tag_id_to_bundles;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;
    std::string map_frame_;
    bool publish_tf_ = true;
    std::string frame_prefix_;
    std::string image_transport_;
    std::string qos_profile_;

    std::function<void(apriltag_family_t*)> tf_destructor;

    // configured
    rclcpp_lifecycle::LifecyclePublisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_pose_array;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    // active
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    // image_transport before 6.4 (Humble) takes only an rclcpp::Node, so the CameraSubscriber
    // lives on a helper node "<name>_camera" in this node's namespace, spun on camera_thread_.
    rclcpp::Node::SharedPtr camera_node_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> camera_executor_;
    std::thread camera_thread_;
    std::atomic<bool> camera_running_{false};
    image_transport::CameraSubscriber sub_cam;

    pose_estimation_f estimate_pose = nullptr;

    void configureDetector();

    void startCamera();

    void stopCamera();

    void releaseActive();

    void releaseConfigured();

    template<typename T>
    T declare(const std::string& name, const T& default_value, const rcl_interfaces::msg::ParameterDescriptor& descriptor);

    void onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);

    void getTagInfoFromDatabase(std::unordered_map<int, std::string>& tag_frames, std::unordered_map<int, double>& tag_sizes);
};
