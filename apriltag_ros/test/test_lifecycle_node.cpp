// AprilTagLifecycleNode (MonKey fork): the camera subscription and the TF listener exist only
// while ACTIVE, and the node has AprilTagNode's parameters, topics and output.
//
// The checks read the ROS graph. CMakeLists runs this test under rmw_fastrtps_cpp, which needs
// no router, on a ROS_DOMAIN_ID of its own, so no other process shares the graph.

#include "AprilTagLifecycleNode.hpp"

#include <apriltag.h>
#include <class_loader/class_loader.hpp>
#include <gtest/gtest.h>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp_components/node_factory.hpp>
#include <tag36h11.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <map>

using namespace std::chrono_literals;
using lifecycle_msgs::msg::State;

namespace {

bool waitFor(const std::function<bool()>& condition, const std::chrono::milliseconds timeout = 5000ms)
{
    const auto end = std::chrono::steady_clock::now() + timeout;
    while(std::chrono::steady_clock::now() < end) {
        if(condition()) { return true; }
        std::this_thread::sleep_for(10ms);
    }
    return condition();
}

// true when the condition holds for the whole duration
bool holds(const std::function<bool()>& condition, const std::chrono::milliseconds duration = 500ms)
{
    const auto end = std::chrono::steady_clock::now() + duration;
    while(std::chrono::steady_clock::now() < end) {
        if(!condition()) { return false; }
        std::this_thread::sleep_for(10ms);
    }
    return condition();
}

std::string uniqueNamespace(const std::string& name)
{
    return "/apriltag_lifecycle_test_" + std::to_string(getpid()) + "/" + name;
}

std::vector<std::string> nodeArguments(const std::string& ns, const std::vector<std::string>& remaps)
{
    std::vector<std::string> arguments{"--ros-args", "-r", "__ns:=" + ns};
    for(const std::string& remap : remaps) {
        arguments.push_back("-r");
        arguments.push_back(remap);
    }
    return arguments;
}

// AprilTagNode has no header: load its component from the library, as a container does.
// Keep the loader alive longer than the nodes it creates.
class AprilTagNodeLoader {
public:
    AprilTagNodeLoader() : loader(APRILTAG_NODE_LIBRARY) {}

    rclcpp_components::NodeInstanceWrapper create(const rclcpp::NodeOptions& options)
    {
        factory = loader.createInstance<rclcpp_components::NodeFactory>("rclcpp_components::NodeFactoryTemplate<AprilTagNode>");
        return factory->create_node_instance(options);
    }

private:
    class_loader::ClassLoader loader;
    std::shared_ptr<rclcpp_components::NodeFactory> factory;
};

// spins an executor on a thread until destroyed
class Spinner {
public:
    explicit Spinner(const std::vector<rclcpp::node_interfaces::NodeBaseInterface::SharedPtr>& nodes)
    {
        for(const auto& node : nodes) { executor.add_node(node); }
        thread = std::thread([this]() {
            while(running && rclcpp::ok()) { executor.spin_once(100ms); }
        });
    }

    ~Spinner()
    {
        running = false;
        executor.cancel();
        thread.join();
    }

private:
    rclcpp::executors::SingleThreadedExecutor executor;
    std::atomic<bool> running{true};
    std::thread thread;
};

// topic -> types of a node's subscriptions or publishers, topics below node_ns made relative
std::map<std::string, std::vector<std::string>>
endpointsByNode(const rclcpp::Node::SharedPtr& probe, const bool subscriptions, const std::string& name, const std::string& node_ns)
{
    std::map<std::string, std::vector<std::string>> endpoints;
    try {
        const auto graph = probe->get_node_graph_interface();
        const auto topics = subscriptions ? graph->get_subscriber_names_and_types_by_node(name, node_ns)
                                          : graph->get_publisher_names_and_types_by_node(name, node_ns);
        for(const auto& topic : topics) {
            const bool below = topic.first.compare(0, node_ns.size() + 1, node_ns + "/") == 0;
            endpoints[below ? topic.first.substr(node_ns.size()) : topic.first] = topic.second;
        }
    }
    catch(const std::exception&) {
        // node not discovered yet
    }
    return endpoints;
}

std::string toString(const std::map<std::string, std::vector<std::string>>& endpoints)
{
    std::string text;
    for(const auto& endpoint : endpoints) {
        text += endpoint.first + " [";
        for(const std::string& type : endpoint.second) { text += type + " "; }
        text += "] ";
    }
    return text;
}

constexpr int kTagId = 5;

// a 36h11 tag, 12 px per cell, off-centre on a white 320x240 image
sensor_msgs::msg::Image tagImage()
{
    apriltag_family_t* const family = tag36h11_create();
    image_u8_t* const tag = apriltag_to_image(family, kTagId);
    const int scale = 12;
    const int x0 = 110;
    const int y0 = 50;

    sensor_msgs::msg::Image image;
    image.header.frame_id = "camera";
    image.width = 320;
    image.height = 240;
    image.encoding = "mono8";
    image.step = image.width;
    image.data.assign(image.step * image.height, 255);
    for(int y = 0; y < tag->height * scale; y++) {
        for(int x = 0; x < tag->width * scale; x++) {
            image.data[(y0 + y) * image.step + x0 + x] = tag->buf[(y / scale) * tag->stride + x / scale];
        }
    }

    image_u8_destroy(tag);
    tag36h11_destroy(family);
    return image;
}

sensor_msgs::msg::CameraInfo cameraInfo()
{
    sensor_msgs::msg::CameraInfo info;
    info.header.frame_id = "camera";
    info.width = 320;
    info.height = 240;
    info.distortion_model = "plumb_bob";
    info.d = {0, 0, 0, 0, 0};
    info.k = {300, 0, 160, 0, 300, 120, 0, 0, 1};
    info.p = {300, 0, 160, 0, 0, 300, 120, 0, 0, 0, 1, 0};
    return info;
}

} // namespace


TEST(AprilTagLifecycleNode, CameraSubscriptionOnlyWhileActive)
{
    const std::string ns = uniqueNamespace("lifecycle");
    rclcpp::NodeOptions options;
    // ECR's pattern: a relative image topic, camera_info taken from its directory. The rule on
    // the derived camera_info topic is one AprilTagNode's own subscription obeys.
    options.arguments(nodeArguments(ns, {"image_rect:=front/cam/image", "front/cam/camera_info:=front/cam/info"}));
    const auto node = std::make_shared<AprilTagLifecycleNode>(options);
    const auto probe = std::make_shared<rclcpp::Node>("probe", ns);

    const std::string image_topic = ns + "/front/cam/image";
    const std::string info_topic = ns + "/front/cam/info";
    // nothing else on this test's domain subscribes /tf: a subscriber there is the TF listener
    const auto idle = [&]() {
        return probe->count_subscribers(image_topic) == 0u && probe->count_subscribers(info_topic) == 0u &&
               probe->count_subscribers("/tf") == 0u;
    };
    const auto subscribed = [&]() {
        return probe->count_subscribers(image_topic) == 1u && probe->count_subscribers(info_topic) == 1u &&
               probe->count_subscribers("/tf") == 1u;
    };

    ASSERT_EQ(node->get_current_state().id(), State::PRIMARY_STATE_UNCONFIGURED);
    EXPECT_FALSE(node->has_parameter("family"));
    EXPECT_TRUE(holds(idle));

    // configure: parameters and publishers, no camera subscription, no TF listener
    ASSERT_EQ(node->configure().id(), State::PRIMARY_STATE_INACTIVE);
    EXPECT_TRUE(node->has_parameter("family"));
    EXPECT_TRUE(node->has_parameter("detector.threads"));
    EXPECT_TRUE(waitFor([&]() {
        return probe->count_publishers(ns + "/detections") == 1u && probe->count_publishers(ns + "/pose_array") == 1u;
    }));
    EXPECT_TRUE(holds(idle));

    // the dock leg activates and deactivates once per dropoff
    for(int cycle = 0; cycle < 2; cycle++) {
        ASSERT_EQ(node->activate().id(), State::PRIMARY_STATE_ACTIVE);
        EXPECT_TRUE(waitFor(subscribed)) << "cycle " << cycle;
        EXPECT_EQ(probe->count_subscribers(ns + "/front/cam/camera_info"), 0u);

        ASSERT_EQ(node->deactivate().id(), State::PRIMARY_STATE_INACTIVE);
        EXPECT_TRUE(waitFor(idle)) << "cycle " << cycle;
        EXPECT_TRUE(holds(idle)) << "cycle " << cycle;
    }

    // cleanup keeps the parameters declared, a second configure reads them back
    ASSERT_TRUE(node->set_parameter(rclcpp::Parameter("max_hamming", 2)).successful);
    ASSERT_EQ(node->cleanup().id(), State::PRIMARY_STATE_UNCONFIGURED);
    EXPECT_TRUE(holds(idle));
    ASSERT_EQ(node->configure().id(), State::PRIMARY_STATE_INACTIVE);
    EXPECT_EQ(node->get_parameter("max_hamming").as_int(), 2);
    ASSERT_EQ(node->activate().id(), State::PRIMARY_STATE_ACTIVE);
    EXPECT_TRUE(waitFor(subscribed));

    ASSERT_EQ(node->shutdown().id(), State::PRIMARY_STATE_FINALIZED);
    EXPECT_TRUE(waitFor(idle));
}

TEST(AprilTagLifecycleNode, SameParametersAsAprilTagNode)
{
    const std::string ns = uniqueNamespace("parameters");
    const auto make_options = [&](const std::string& name) {
        rclcpp::NodeOptions options;
        options.arguments(nodeArguments(ns + "/" + name, {}));
        options.parameter_overrides({
            rclcpp::Parameter("tag.ids", std::vector<int64_t>{1, 2}),
            rclcpp::Parameter("tag.frames", std::vector<std::string>{"one", "two"}),
            rclcpp::Parameter("tag_bundles.bundle_names", std::vector<std::string>{"bundle"}),
            rclcpp::Parameter("tag_bundles.bundle.ids", std::vector<int64_t>{3}),
            rclcpp::Parameter("detector.threads", 2),
            rclcpp::Parameter("max_hamming", 1),
        });
        return options;
    };
    AprilTagNodeLoader loader;
    auto reference = loader.create(make_options("reference"));
    const auto node = std::make_shared<AprilTagLifecycleNode>(make_options("lifecycle"));
    ASSERT_EQ(node->configure().id(), State::PRIMARY_STATE_INACTIVE);

    // AprilTagNode's parameters through its parameter services
    const auto tester = std::make_shared<rclcpp::Node>("tester", ns);
    const auto client = std::make_shared<rclcpp::AsyncParametersClient>(tester, ns + "/reference/apriltag");
    const Spinner spinner({reference.get_node_base_interface(), tester->get_node_base_interface()});
    ASSERT_TRUE(client->wait_for_service(5s));

    const auto listed = client->list_parameters({}, rcl_interfaces::srv::ListParameters::Request::DEPTH_RECURSIVE);
    ASSERT_EQ(listed.wait_for(5s), std::future_status::ready);
    std::vector<std::string> names = listed.get().names;
    // rclcpp::Node declares QoS overrides for its parameter event publisher, a Humble
    // rclcpp_lifecycle::LifecycleNode does not: framework parameters, not this package's
    const std::string parameter_events = "qos_overrides./parameter_events.";
    names.erase(std::remove_if(names.begin(), names.end(), [&](const std::string& name) { return name.compare(0, parameter_events.size(), parameter_events) == 0; }),
                names.end());
    std::vector<std::string> lifecycle_names = node->list_parameters({}, rcl_interfaces::srv::ListParameters::Request::DEPTH_RECURSIVE).names;
    std::sort(names.begin(), names.end());
    std::sort(lifecycle_names.begin(), lifecycle_names.end());
    EXPECT_EQ(names, lifecycle_names);
    EXPECT_EQ(std::count(names.begin(), names.end(), "tag_bundles.bundle.3.transform"), 1);

    const auto described = client->describe_parameters(names);
    ASSERT_EQ(described.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(described.get(), node->describe_parameters(names));
    const auto values = client->get_parameters(names);
    ASSERT_EQ(values.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(values.get(), node->get_parameters(names));
}

TEST(AprilTagLifecycleNode, SameTopicsAsAprilTagNode)
{
    const std::string ns = uniqueNamespace("topics");
    // a relative image topic, a rule on the derived camera_info topic, and a camera_info rule
    // that AprilTagNode does not apply because its derived camera_info topic differs
    const std::vector<std::string> remaps{"image_rect:=front/cam/image", "front/cam/camera_info:=front/cam/info",
                                          "camera_info:=unused/camera_info", "detections:=april_tag/detections",
                                          "pose_array:=april_tag/pose"};
    AprilTagNodeLoader loader;
    rclcpp::NodeOptions reference_options;
    reference_options.arguments(nodeArguments(ns + "/reference", remaps));
    auto reference = loader.create(reference_options);

    rclcpp::NodeOptions options;
    options.arguments(nodeArguments(ns + "/lifecycle", remaps));
    const auto node = std::make_shared<AprilTagLifecycleNode>(options);
    ASSERT_EQ(node->configure().id(), State::PRIMARY_STATE_INACTIVE);
    ASSERT_EQ(node->activate().id(), State::PRIMARY_STATE_ACTIVE);

    const auto probe = std::make_shared<rclcpp::Node>("probe", ns);
    // every rclcpp node subscribes /parameter_events (time source)
    const std::map<std::string, std::vector<std::string>> subscriptions{
        {"/front/cam/image", {"sensor_msgs/msg/Image"}},
        {"/front/cam/info", {"sensor_msgs/msg/CameraInfo"}},
        {"/parameter_events", {"rcl_interfaces/msg/ParameterEvent"}},
    };
    // the camera subscription of the lifecycle node lives on its helper node <name>_camera
    EXPECT_TRUE(waitFor([&]() { return endpointsByNode(probe, true, "apriltag", ns + "/reference") == subscriptions; }))
        << toString(endpointsByNode(probe, true, "apriltag", ns + "/reference"));
    EXPECT_TRUE(waitFor([&]() { return endpointsByNode(probe, true, "apriltag_camera", ns + "/lifecycle") == subscriptions; }))
        << toString(endpointsByNode(probe, true, "apriltag_camera", ns + "/lifecycle"));
    EXPECT_EQ(endpointsByNode(probe, true, "apriltag", ns + "/reference"),
              endpointsByNode(probe, true, "apriltag_camera", ns + "/lifecycle"));

    // same publishers, apart from the lifecycle transition events
    const auto has_outputs = [&](const std::string& node_ns) {
        return endpointsByNode(probe, false, "apriltag", node_ns).count("/april_tag/pose") == 1;
    };
    EXPECT_TRUE(waitFor([&]() { return has_outputs(ns + "/reference") && has_outputs(ns + "/lifecycle"); }));
    auto publishers = endpointsByNode(probe, false, "apriltag", ns + "/lifecycle");
    EXPECT_EQ(publishers.erase("/apriltag/transition_event"), 1u);
    EXPECT_EQ(endpointsByNode(probe, false, "apriltag", ns + "/reference"), publishers);
    EXPECT_EQ(publishers.count("/april_tag/detections"), 1u);
    EXPECT_EQ(publishers.count("/tf"), 1u);
}

TEST(AprilTagLifecycleNode, SameOutputAsAprilTagNode)
{
    const std::string ns = uniqueNamespace("output");
    const std::string image_topic = ns + "/camera/image";
    const std::vector<std::string> names{"reference", "lifecycle"};
    const auto make_options = [&](const std::string& name) {
        rclcpp::NodeOptions options;
        options.arguments(nodeArguments(ns + "/" + name, {"image_rect:=" + image_topic}));
        // the prefix tells the two TF outputs apart
        options.parameter_overrides({rclcpp::Parameter("frame_prefix", name + "/"), rclcpp::Parameter("size", 0.16)});
        return options;
    };

    AprilTagNodeLoader loader;
    auto reference = loader.create(make_options("reference"));
    const auto node = std::make_shared<AprilTagLifecycleNode>(make_options("lifecycle"));
    ASSERT_EQ(node->configure().id(), State::PRIMARY_STATE_INACTIVE);
    ASSERT_EQ(node->activate().id(), State::PRIMARY_STATE_ACTIVE);

    const auto tester = std::make_shared<rclcpp::Node>("tester", ns);

    // camera optical frame -> map, for pose_array
    tf2_ros::StaticTransformBroadcaster static_broadcaster(tester);
    geometry_msgs::msg::TransformStamped camera_to_map;
    camera_to_map.header.frame_id = "map";
    camera_to_map.child_frame_id = "camera";
    camera_to_map.transform.translation.x = 1.0;
    camera_to_map.transform.translation.y = 2.0;
    camera_to_map.transform.translation.z = 0.5;
    camera_to_map.transform.rotation.x = -0.5;
    camera_to_map.transform.rotation.y = 0.5;
    camera_to_map.transform.rotation.z = -0.5;
    camera_to_map.transform.rotation.w = 0.5;
    static_broadcaster.sendTransform(camera_to_map);

    // outputs per node and image stamp
    std::mutex mutex;
    std::map<std::string, std::map<int64_t, apriltag_msgs::msg::AprilTagDetectionArray>> detections;
    std::map<std::string, std::map<int64_t, geometry_msgs::msg::PoseArray>> pose_arrays;
    std::map<std::string, std::map<int64_t, std::map<std::string, geometry_msgs::msg::TransformStamped>>> transforms;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;
    for(const std::string& name : names) {
        subscriptions.push_back(tester->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>(
            ns + "/" + name + "/detections", 10, [&, name](const apriltag_msgs::msg::AprilTagDetectionArray::ConstSharedPtr msg) {
                const std::lock_guard<std::mutex> lock(mutex);
                detections[name][rclcpp::Time(msg->header.stamp).nanoseconds()] = *msg;
            }));
        subscriptions.push_back(tester->create_subscription<geometry_msgs::msg::PoseArray>(
            ns + "/" + name + "/pose_array", 10, [&, name](const geometry_msgs::msg::PoseArray::ConstSharedPtr msg) {
                const std::lock_guard<std::mutex> lock(mutex);
                pose_arrays[name][rclcpp::Time(msg->header.stamp).nanoseconds()] = *msg;
            }));
    }
    subscriptions.push_back(tester->create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", 100, [&](const tf2_msgs::msg::TFMessage::ConstSharedPtr msg) {
            const std::lock_guard<std::mutex> lock(mutex);
            for(const auto& transform : msg->transforms) {
                for(const std::string& name : names) {
                    if(transform.child_frame_id.compare(0, name.size() + 1, name + "/") == 0) {
                        geometry_msgs::msg::TransformStamped unprefixed = transform;
                        unprefixed.child_frame_id = transform.child_frame_id.substr(name.size() + 1);
                        transforms[name][rclcpp::Time(transform.header.stamp).nanoseconds()][unprefixed.child_frame_id] = unprefixed;
                    }
                }
            }
        }));

    const auto pub_image = tester->create_publisher<sensor_msgs::msg::Image>(image_topic, 10);
    const auto pub_info = tester->create_publisher<sensor_msgs::msg::CameraInfo>(ns + "/camera/camera_info", 10);

    const Spinner spinner({reference.get_node_base_interface(), tester->get_node_base_interface()});

    ASSERT_TRUE(waitFor([&]() { return pub_image->get_subscription_count() == 2u && pub_info->get_subscription_count() == 2u; }));
    ASSERT_TRUE(waitFor([&]() {
        for(const auto& subscription : subscriptions) {
            if(subscription->get_publisher_count() == 0u) { return false; }
        }
        return true;
    }));

    sensor_msgs::msg::Image image = tagImage();
    sensor_msgs::msg::CameraInfo info = cameraInfo();
    // publish frames until both nodes produced every output for one stamp (TF listeners warm up)
    const auto complete = [&](const int64_t stamp) {
        const std::lock_guard<std::mutex> lock(mutex);
        for(const std::string& name : names) {
            if(!detections[name].count(stamp) || !pose_arrays[name].count(stamp) || !transforms[name].count(stamp) ||
               pose_arrays[name][stamp].poses.empty()) {
                return false;
            }
        }
        return true;
    };
    int64_t stamp = 0;
    bool done = false;
    for(int frame = 1; frame <= 30 && !done; frame++) {
        image.header.stamp = info.header.stamp = rclcpp::Time(frame, 0);
        stamp = rclcpp::Time(image.header.stamp).nanoseconds();
        pub_info->publish(info);
        pub_image->publish(image);
        done = waitFor([&]() { return complete(stamp); }, 1000ms);
    }
    ASSERT_TRUE(done);

    const std::lock_guard<std::mutex> lock(mutex);
    ASSERT_EQ(detections["reference"][stamp].detections.size(), 1u);
    EXPECT_EQ(detections["reference"][stamp].detections[0].id, kTagId);
    EXPECT_EQ(detections["reference"][stamp], detections["lifecycle"][stamp]);
    ASSERT_EQ(pose_arrays["reference"][stamp].poses.size(), 1u);
    EXPECT_EQ(pose_arrays["reference"][stamp].header.frame_id, "map");
    EXPECT_EQ(pose_arrays["reference"][stamp], pose_arrays["lifecycle"][stamp]);
    ASSERT_EQ(transforms["reference"][stamp].count("tag36h11:" + std::to_string(kTagId)), 1u);
    EXPECT_EQ(transforms["reference"][stamp], transforms["lifecycle"][stamp]);

    // and the output is the expected one: 0.16 m over 8 cells of 12 px at f = 300 px is 0.5 m away,
    // the centre 10 px right of and 10 px above the principal point
    const auto& centre = detections["lifecycle"][stamp].detections[0].centre;
    EXPECT_NEAR(centre.x, 170.0, 0.5);
    EXPECT_NEAR(centre.y, 110.0, 0.5);
    EXPECT_NEAR(transforms["lifecycle"][stamp].begin()->second.transform.translation.z, 0.5, 1e-3);
    const auto& position = pose_arrays["lifecycle"][stamp].poses[0].position;
    EXPECT_NEAR(position.x, 1.5, 1e-3);
    EXPECT_NEAR(position.y, 2.0 - 0.5 / 30, 1e-3);
    EXPECT_NEAR(position.z, 0.5 + 0.5 / 30, 1e-3);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    const int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
