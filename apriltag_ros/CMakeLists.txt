cmake_minimum_required(VERSION 3.16)

project(apriltag_ros)

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Werror -Wall -Wextra -Wpedantic)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options("-Wl,-z,relro,-z,now,-z,defs")
  endif()
endif()

option(ASAN "use AddressSanitizer to detect memory issues" OFF)

if(ASAN)
    set(ASAN_FLAGS "\
        -fsanitize=address \
        -fsanitize=bool \
        -fsanitize=bounds \
        -fsanitize=enum \
        -fsanitize=float-cast-overflow \
        -fsanitize=float-divide-by-zero \
        -fsanitize=nonnull-attribute \
        -fsanitize=returns-nonnull-attribute \
        -fsanitize=signed-integer-overflow \
        -fsanitize=undefined \
        -fsanitize=vla-bound \
        -fno-sanitize=alignment \
        -fsanitize=leak \
        -fsanitize=object-size \
    ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
endif()

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(apriltag_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core calib3d)
find_package(apriltag 3.2 REQUIRED)
find_package(SQLite3 REQUIRED)

if(cv_bridge_VERSION VERSION_GREATER_EQUAL 3.3.0)
    add_compile_definitions(cv_bridge_HPP)
endif()

if(image_transport_VERSION VERSION_GREATER_EQUAL 6.3)
    add_compile_definitions(image_transport_QoS)
endif()

if(image_transport_VERSION VERSION_GREATER_EQUAL 6.4)
    add_compile_definitions(image_transport_NODE_INTERFACE)
endif()

if(tf2_ros_VERSION VERSION_GREATER_EQUAL 0.45)
    add_compile_definitions(tf2_ros_NODE_INTERFACE)
endif()


# database of tag functions
add_library(tags STATIC
    src/tag_functions.cpp
)
target_link_libraries(tags
    apriltag::apriltag
)
set_property(TARGET tags PROPERTY
    POSITION_INDEPENDENT_CODE ON
)


# conversion functions as template specialisation
add_library(conversion STATIC
    src/conversion.cpp
)
target_link_libraries(conversion
  PUBLIC
    apriltag::apriltag
    Eigen3::Eigen
    opencv_core
    ${geometry_msgs_TARGETS}
    tf2::tf2
)
set_property(TARGET conversion PROPERTY
    POSITION_INDEPENDENT_CODE ON
)


# pose estimation methods
add_library(pose_estimation STATIC
    src/pose_estimation.cpp
)
target_link_libraries(pose_estimation
  PUBLIC
    apriltag::apriltag
    Eigen3::Eigen
    opencv_calib3d
    conversion
    ${geometry_msgs_TARGETS}
    tf2::tf2
)
set_property(TARGET pose_estimation PROPERTY
    POSITION_INDEPENDENT_CODE ON
)


# composable node
add_library(AprilTagNode SHARED
    src/AprilTagNode.cpp
)
target_link_libraries(AprilTagNode
  PUBLIC
    apriltag::apriltag
    tags
    pose_estimation
    ${apriltag_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    cv_bridge::cv_bridge
    image_transport::image_transport
    rclcpp::rclcpp
    rclcpp_components::component
    rclcpp_components::component_manager
    sensor_msgs::sensor_msgs_library
    tf2_ros::tf2_ros
    tf2_geometry_msgs::tf2_geometry_msgs
)
rclcpp_components_register_node(AprilTagNode
    PLUGIN "AprilTagNode"
    EXECUTABLE "apriltag_node"
)
target_link_libraries(AprilTagNode PRIVATE SQLite::SQLite3)


# lifecycle variant of the composable node (MonKey fork)
add_library(AprilTagLifecycleNode SHARED
    src/AprilTagLifecycleNode.cpp
)
target_link_libraries(AprilTagLifecycleNode
  PUBLIC
    apriltag::apriltag
    tags
    pose_estimation
    ${apriltag_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    cv_bridge::cv_bridge
    image_transport::image_transport
    rclcpp::rclcpp
    rclcpp_components::component
    rclcpp_lifecycle::rclcpp_lifecycle
    sensor_msgs::sensor_msgs_library
    tf2_ros::tf2_ros
    tf2_geometry_msgs::tf2_geometry_msgs
)
target_link_libraries(AprilTagLifecycleNode PRIVATE SQLite::SQLite3)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Keep the globals of the static libraries (tag_fun, pose_estimation_methods) local to this
  # library, so they cannot interpose with AprilTagNode's copies when both load into one process.
  target_link_options(AprilTagLifecycleNode PRIVATE "LINKER:--exclude-libs,ALL")
endif()
rclcpp_components_register_node(AprilTagLifecycleNode
    PLUGIN "AprilTagLifecycleNode"
    EXECUTABLE "apriltag_lifecycle_node"
)
install(TARGETS AprilTagLifecycleNode
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

ament_environment_hooks(${ament_cmake_package_templates_ENVIRONMENT_HOOK_LIBRARY_PATH})

install(TARGETS AprilTagNode
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY cfg
    DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  set(ament_cmake_clang_format_CONFIG_FILE "${CMAKE_SOURCE_DIR}/.clang-format")
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # AprilTagLifecycleNode: camera subscription only while ACTIVE, and parity with AprilTagNode.
  # Graph checks run under rmw_fastrtps_cpp, which needs no router, on a domain of their own.
  find_package(ament_cmake_gtest REQUIRED)
  find_package(class_loader REQUIRED)
  ament_add_gtest(test_lifecycle_node test/test_lifecycle_node.cpp
    TIMEOUT 120
    ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=87
  )
  if(TARGET test_lifecycle_node)
    target_include_directories(test_lifecycle_node PRIVATE src)
    target_compile_definitions(test_lifecycle_node PRIVATE
      APRILTAG_NODE_LIBRARY="$<TARGET_FILE:AprilTagNode>"
    )
    target_link_libraries(test_lifecycle_node
      AprilTagLifecycleNode
      class_loader::class_loader
    )
  endif()
endif()

ament_package()
