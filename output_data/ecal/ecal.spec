%global forgeurl https://github.com/odra/ecal
%global branch rpm
%global debug_package %{nil}

%forgemeta -i

Name:    ecal
Version: 5.11.1
Release: 1%{?dist}
Summary: Eclipse's enhanced Communication Abstraction Layer (eCAL) 
URL:     %{forgeurl}
Source:  %{forgesource}
License: Apache-2.0

BuildRequires: g++
BuildRequires: cmake
BuildRequires: git
BuildRequires: doxygen
BuildRequires: graphviz 
BuildRequires: protobuf
BuildRequires: protobuf-devel
BuildRequires: asio-devel
BuildRequires: tclap
BuildRequires: simpleini
BuildRequires: spdlog
BuildRequires: spdlog-devel
BuildRequires: gtest
BuildRequires: gtest-devel
BuildRequires: fineftp-server
BuildRequires: tinyxml2
BuildRequires: tinyxml2-devel
BuildRequires: curl-devel
BuildRequires: zlib
BuildRequires: zlib-devel
BuildRequires: libssh2-devel
BuildRequires: hdf5-devel
BuildRequires: termcolor-devel
BuildRequires: recycle
BuildRequires: tcp_pubsub
BuildRequires: qwt-devel
BuildRequires: qt5-qtbase
BuildRequires: qt5-qtbase-devel
BuildRequires: qt5-qtsvg
BuildRequires: qt5-qtsvg-devel
BuildRequires: capnproto
BuildRequires: capnproto-devel
BuildRequires: capnproto-libs

%description
iThe enhanced Communication Abstraction Layer (eCAL) is a middleware that enables scalable, high performance interprocess communication on a single computer node or between different nodes in a computer network. eCAL uses a publish - subscribe pattern to automatically connect different nodes in the network.

%prep
%forgesetup

%build
cmake \
  -DHAS_HDF5=ON \
  -DHAS_QT5=ON \
  -DHAS_CURL=ON \
  -DHAS_CAPNPROTO=ON \
  -DHAS_FTXUI=OFF \
  -DBUILD_DOCS=OFF \
  -DBUILD_APPS=OFF \
  -DBUILD_SAMPLES=OFF \
  -DBUILD_TIME=OFF \
  -DBUILD_PY_BINDING=OFF \
  -DBUILD_STANDALONE_PY_WHEEL=OFF \
  -DBUILD_CSHARP_BINDING=OFF \
  -DBUILD_ECAL_TESTS=OFF \
  -DECAL_LAYER_ICEORYX=OFF \
  -DECAL_INCLUDE_PY_SAMPLES=OFF \
  -DECAL_INSTALL_SAMPLE_SOURCES=OFF \
  -DECAL_JOIN_MULTICAST_TWICE=OFF \
  -DECAL_NPCAP_SUPPORT=OFF \
  -DECAL_THIRDPARTY_BUILD_CMAKE_FUNCTIONS=ON \
  -DECAL_THIRDPARTY_BUILD_PROTOBUF=OFF \
  -DECAL_THIRDPARTY_BUILD_SPDLOG=OFF \
  -DECAL_THIRDPARTY_BUILD_TINYXML2=OFF \
  -DECAL_THIRDPARTY_BUILD_FINEFTP=OFF \
  -DECAL_THIRDPARTY_BUILD_CURL=OFF \
  -DECAL_THIRDPARTY_BUILD_GTEST=OFF \
  -DECAL_THIRDPARTY_BUILD_HDF5=OFF \
  -DECAL_THIRDPARTY_BUILD_RECYCLE=OFF \
  -DECAL_THIRDPARTY_BUILD_TCP_PUBSUB=OFF \
  -DECAL_THIRDPARTY_BUILD_QWT=OFF \
  -DECAL_THIRDPARTY_BUILD_FTXUI=OFF \
  -DECAL_THIRDPARTY_BUILD_TERMCOLOR=OFF \
  -DECAL_THIRDPARTY_BUILD_YAML-CPP=OFF \
  -Dasio_INCLUDE_DIR=%{_includedir}/asio \
  -Dtclap_INCLUDE_DIR=%{_includedir}/tclap \
  -Dtcp_pubsub_INCLUDE_DIR=%{_includedir}/tcp_pubsub \
  -Dsimpleini_INCLUDE_DIR=/usr/local/include \
  -DCMAKE_INSTALL_PREFIX=_build \
  -Dtcp_pubsub_DIR=%{_libdir}/cmake/tcp_pubsub
make

%install
make install
# install files
install -D  _build/etc/ecal/ecal.ini %{buildroot}%{_sysconfdir}/ecal/ecal.ini
install -D  _build/bin/ecal_process_stub-0.0.0 %{buildroot}%{_bindir}/ecal_process_stub-0.0.0
install -D  _build/include/ecalhdf5/eh5_defs.h %{buildroot}%{_includedir}/ecalhdf5/eh5_defs.h
install -D  _build/include/ecalhdf5/eh5_meas.h %{buildroot}%{_includedir}/ecalhdf5/eh5_meas.h
install -D  _build/include/ecalhdf5/eh5_types.h %{buildroot}%{_includedir}/ecalhdf5/eh5_types.h
install -D  _build/include/ecal/measurement/imeasurement.h %{buildroot}%{_includedir}/ecal/measurement/imeasurement.h
install -D  _build/include/ecal/measurement/measurement.h %{buildroot}%{_includedir}/ecal/measurement/measurement.h
install -D  _build/include/ecal/measurement/omeasurement.h %{buildroot}%{_includedir}/ecal/measurement/omeasurement.h
install -D  _build/include/ecal/rec/concurrent_status_interface.h %{buildroot}%{_includedir}/ecal/rec/concurrent_status_interface.h
install -D  _build/include/ecal/rec/recorder_impl_base.h %{buildroot}%{_includedir}/ecal/rec/recorder_impl_base.h
install -D  _build/include/ecal/rec/recorder_impl_base_types.h %{buildroot}%{_includedir}/ecal/rec/recorder_impl_base_types.h
install -D  _build/include/ecal/mon/plugin.h %{buildroot}%{_includedir}/ecal/mon/plugin.h
install -D  _build/include/ecal/mon/plugin_interface.h %{buildroot}%{_includedir}/ecal/mon/plugin_interface.h
install -D  _build/include/ecal/mon/plugin_widget_interface.h %{buildroot}%{_includedir}/ecal/mon/plugin_widget_interface.h
install -D  _build/include/ecal/cimpl/ecal_callback_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_callback_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_client_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_client_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_core_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_core_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_event_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_event_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_init_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_init_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_log_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_log_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_monitoring_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_monitoring_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_process_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_process_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_proto_dyn_json_subscriber_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_proto_dyn_json_subscriber_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_publisher_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_publisher_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_qos_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_qos_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_server_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_server_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_service_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_service_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_service_info_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_service_info_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_subscriber_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_subscriber_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_time_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_time_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_timer_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_timer_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_tlayer_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_tlayer_cimpl.h
install -D  _build/include/ecal/cimpl/ecal_util_cimpl.h %{buildroot}%{_includedir}/ecal/cimpl/ecal_util_cimpl.h
install -D  _build/include/ecal/ecal.h %{buildroot}%{_includedir}/ecal/ecal.h
install -D  _build/include/ecal/ecal_callback.h %{buildroot}%{_includedir}/ecal/ecal_callback.h
install -D  _build/include/ecal/ecal_clang.h %{buildroot}%{_includedir}/ecal/ecal_clang.h
install -D  _build/include/ecal/ecal_client.h %{buildroot}%{_includedir}/ecal/ecal_client.h
install -D  _build/include/ecal/ecal_config.h %{buildroot}%{_includedir}/ecal/ecal_config.h
install -D  _build/include/ecal/ecal_core.h %{buildroot}%{_includedir}/ecal/ecal_core.h
install -D  _build/include/ecal/ecal_event.h %{buildroot}%{_includedir}/ecal/ecal_event.h
install -D  _build/include/ecal/ecal_eventhandle.h %{buildroot}%{_includedir}/ecal/ecal_eventhandle.h
install -D  _build/include/ecal/ecal_init.h %{buildroot}%{_includedir}/ecal/ecal_init.h
install -D  _build/include/ecal/ecal_log.h %{buildroot}%{_includedir}/ecal/ecal_log.h
install -D  _build/include/ecal/ecal_log_level.h %{buildroot}%{_includedir}/ecal/ecal_log_level.h
install -D  _build/include/ecal/ecal_monitoring.h %{buildroot}%{_includedir}/ecal/ecal_monitoring.h
install -D  _build/include/ecal/ecal_os.h %{buildroot}%{_includedir}/ecal/ecal_os.h
install -D  _build/include/ecal/ecal_process.h %{buildroot}%{_includedir}/ecal/ecal_process.h
install -D  _build/include/ecal/ecal_process_mode.h %{buildroot}%{_includedir}/ecal/ecal_process_mode.h
install -D  _build/include/ecal/ecal_process_severity.h %{buildroot}%{_includedir}/ecal/ecal_process_severity.h
install -D  _build/include/ecal/ecal_publisher.h %{buildroot}%{_includedir}/ecal/ecal_publisher.h
install -D  _build/include/ecal/ecal_qos.h %{buildroot}%{_includedir}/ecal/ecal_qos.h
install -D  _build/include/ecal/ecal_server.h %{buildroot}%{_includedir}/ecal/ecal_server.h
install -D  _build/include/ecal/ecal_service.h %{buildroot}%{_includedir}/ecal/ecal_service.h
install -D  _build/include/ecal/ecal_service_info.h %{buildroot}%{_includedir}/ecal/ecal_service_info.h
install -D  _build/include/ecal/ecal_subscriber.h %{buildroot}%{_includedir}/ecal/ecal_subscriber.h
install -D  _build/include/ecal/ecal_time.h %{buildroot}%{_includedir}/ecal/ecal_time.h
install -D  _build/include/ecal/ecal_timed_cb.h %{buildroot}%{_includedir}/ecal/ecal_timed_cb.h
install -D  _build/include/ecal/ecal_timer.h %{buildroot}%{_includedir}/ecal/ecal_timer.h
install -D  _build/include/ecal/ecal_tlayer.h %{buildroot}%{_includedir}/ecal/ecal_tlayer.h
install -D  _build/include/ecal/ecal_util.h %{buildroot}%{_includedir}/ecal/ecal_util.h
install -D  _build/include/ecal/ecalc.h %{buildroot}%{_includedir}/ecal/ecalc.h
install -D  _build/include/ecal/ecalc_types.h %{buildroot}%{_includedir}/ecal/ecalc_types.h
install -D  _build/include/ecal/msg/message.h %{buildroot}%{_includedir}/ecal/msg/message.h
install -D  _build/include/ecal/msg/proto/message.h %{buildroot}%{_includedir}/ecal/msg/proto/message.h
install -D  _build/include/ecal/msg/capnproto/dynamic.h %{buildroot}%{_includedir}/ecal/msg/capnproto/dynamic.h
install -D  _build/include/ecal/msg/capnproto/helper.h %{buildroot}%{_includedir}/ecal/msg/capnproto/helper.h
install -D  _build/include/ecal/msg/capnproto/publisher.h %{buildroot}%{_includedir}/ecal/msg/capnproto/publisher.h
install -D  _build/include/ecal/msg/capnproto/subscriber.h %{buildroot}%{_includedir}/ecal/msg/capnproto/subscriber.h
install -D  _build/include/ecal/msg/dynamic.h %{buildroot}%{_includedir}/ecal/msg/dynamic.h
install -D  _build/include/ecal/msg/flatbuffers/publisher.h %{buildroot}%{_includedir}/ecal/msg/flatbuffers/publisher.h
install -D  _build/include/ecal/msg/flatbuffers/subscriber.h %{buildroot}%{_includedir}/ecal/msg/flatbuffers/subscriber.h
install -D  _build/include/ecal/msg/messagepack/publisher.h %{buildroot}%{_includedir}/ecal/msg/messagepack/publisher.h
install -D  _build/include/ecal/msg/messagepack/subscriber.h %{buildroot}%{_includedir}/ecal/msg/messagepack/subscriber.h
install -D  _build/include/ecal/msg/protobuf/client.h %{buildroot}%{_includedir}/ecal/msg/protobuf/client.h
install -D  _build/include/ecal/msg/protobuf/dynamic_json_subscriber.h %{buildroot}%{_includedir}/ecal/msg/protobuf/dynamic_json_subscriber.h
install -D  _build/include/ecal/msg/protobuf/dynamic_publisher.h %{buildroot}%{_includedir}/ecal/msg/protobuf/dynamic_publisher.h
install -D  _build/include/ecal/msg/protobuf/dynamic_subscriber.h %{buildroot}%{_includedir}/ecal/msg/protobuf/dynamic_subscriber.h
install -D  _build/include/ecal/msg/protobuf/publisher.h %{buildroot}%{_includedir}/ecal/msg/protobuf/publisher.h
install -D  _build/include/ecal/msg/protobuf/server.h %{buildroot}%{_includedir}/ecal/msg/protobuf/server.h
install -D  _build/include/ecal/msg/protobuf/subscriber.h %{buildroot}%{_includedir}/ecal/msg/protobuf/subscriber.h
install -D  _build/include/ecal/msg/publisher.h %{buildroot}%{_includedir}/ecal/msg/publisher.h
install -D  _build/include/ecal/msg/string/message.h %{buildroot}%{_includedir}/ecal/msg/string/message.h
install -D  _build/include/ecal/msg/string/publisher.h %{buildroot}%{_includedir}/ecal/msg/string/publisher.h
install -D  _build/include/ecal/msg/string/subscriber.h %{buildroot}%{_includedir}/ecal/msg/string/subscriber.h
install -D  _build/include/ecal/msg/subscriber.h %{buildroot}%{_includedir}/ecal/msg/subscriber.h
install -D  _build/include/ecal/ecal_defs.h %{buildroot}%{_includedir}/ecal/ecal_defs.h
install -D  _build/include/ecal/app/pb/sys/state.pb.h %{buildroot}%{_includedir}/ecal/app/pb/sys/state.pb.h
install -D  _build/include/ecal/app/pb/sys/client_service.pb.h %{buildroot}%{_includedir}/ecal/app/pb/sys/client_service.pb.h
install -D  _build/include/ecal/app/pb/sys/process.pb.h %{buildroot}%{_includedir}/ecal/app/pb/sys/process.pb.h
install -D  _build/include/ecal/app/pb/sys/service.pb.h %{buildroot}%{_includedir}/ecal/app/pb/sys/service.pb.h
install -D  _build/include/ecal/app/pb/mma/mma.pb.h %{buildroot}%{_includedir}/ecal/app/pb/mma/mma.pb.h
install -D  _build/include/ecal/app/pb/play/service.pb.h %{buildroot}%{_includedir}/ecal/app/pb/play/service.pb.h
install -D  _build/include/ecal/app/pb/play/state.pb.h %{buildroot}%{_includedir}/ecal/app/pb/play/state.pb.h
install -D  _build/include/ecal/app/pb/rec/client_service.pb.h %{buildroot}%{_includedir}/ecal/app/pb/rec/client_service.pb.h
install -D  _build/include/ecal/app/pb/rec/client_state.pb.h %{buildroot}%{_includedir}/ecal/app/pb/rec/client_state.pb.h
install -D  _build/include/ecal/app/pb/rec/server_config.pb.h %{buildroot}%{_includedir}/ecal/app/pb/rec/server_config.pb.h
install -D  _build/include/ecal/app/pb/rec/server_service.pb.h %{buildroot}%{_includedir}/ecal/app/pb/rec/server_service.pb.h
install -D  _build/include/ecal/app/pb/rec/server_state.pb.h %{buildroot}%{_includedir}/ecal/app/pb/rec/server_state.pb.h
install -D  _build/include/ecal/core/pb/topic.pb.h %{buildroot}%{_includedir}/ecal/core/pb/topic.pb.h
install -D  _build/include/ecal/core/pb/ecal.pb.h %{buildroot}%{_includedir}/ecal/core/pb/ecal.pb.h
install -D  _build/include/ecal/core/pb/host.pb.h %{buildroot}%{_includedir}/ecal/core/pb/host.pb.h
install -D  _build/include/ecal/core/pb/layer.pb.h %{buildroot}%{_includedir}/ecal/core/pb/layer.pb.h
install -D  _build/include/ecal/core/pb/monitoring.pb.h %{buildroot}%{_includedir}/ecal/core/pb/monitoring.pb.h
install -D  _build/include/ecal/core/pb/process.pb.h %{buildroot}%{_includedir}/ecal/core/pb/process.pb.h
install -D  _build/include/ecal/core/pb/service.pb.h %{buildroot}%{_includedir}/ecal/core/pb/service.pb.h
install -D  _build/include/ecal/pb/ecal.pb.h %{buildroot}%{_includedir}/ecal/pb/ecal.pb.h
install -D  _build/include/ecal/pb/host.pb.h %{buildroot}%{_includedir}/ecal/pb/host.pb.h
install -D  _build/include/ecal/pb/layer.pb.h %{buildroot}%{_includedir}/ecal/pb/layer.pb.h
install -D  _build/include/ecal/pb/mma/mma.pb.h %{buildroot}%{_includedir}/ecal/pb/mma/mma.pb.h
install -D  _build/include/ecal/pb/monitoring.pb.h %{buildroot}%{_includedir}/ecal/pb/monitoring.pb.h
install -D  _build/include/ecal/pb/play/service.pb.h %{buildroot}%{_includedir}/ecal/pb/play/service.pb.h
install -D  _build/include/ecal/pb/play/state.pb.h %{buildroot}%{_includedir}/ecal/pb/play/state.pb.h
install -D  _build/include/ecal/pb/process.pb.h %{buildroot}%{_includedir}/ecal/pb/process.pb.h
install -D  _build/include/ecal/pb/rec/client_service.pb.h %{buildroot}%{_includedir}/ecal/pb/rec/client_service.pb.h
install -D  _build/include/ecal/pb/rec/client_state.pb.h %{buildroot}%{_includedir}/ecal/pb/rec/client_state.pb.h
install -D  _build/include/ecal/pb/rec/server_config.pb.h %{buildroot}%{_includedir}/ecal/pb/rec/server_config.pb.h
install -D  _build/include/ecal/pb/rec/server_service.pb.h %{buildroot}%{_includedir}/ecal/pb/rec/server_service.pb.h
install -D  _build/include/ecal/pb/rec/server_state.pb.h %{buildroot}%{_includedir}/ecal/pb/rec/server_state.pb.h
install -D  _build/include/ecal/pb/service.pb.h %{buildroot}%{_includedir}/ecal/pb/service.pb.h
install -D  _build/include/ecal/pb/sim_time.pb.h %{buildroot}%{_includedir}/ecal/pb/sim_time.pb.h
install -D  _build/include/ecal/pb/sys/client_service.pb.h %{buildroot}%{_includedir}/ecal/pb/sys/client_service.pb.h
install -D  _build/include/ecal/pb/sys/service.pb.h %{buildroot}%{_includedir}/ecal/pb/sys/service.pb.h
install -D  _build/include/ecal/pb/sys/state.pb.h %{buildroot}%{_includedir}/ecal/pb/sys/state.pb.h
install -D  _build/include/ecal/pb/topic.pb.h %{buildroot}%{_includedir}/ecal/pb/topic.pb.h
install -D  _build/include/ecal/protobuf/ecal_proto_visitor.h %{buildroot}%{_includedir}/ecal/protobuf/ecal_proto_visitor.h
install -D  _build/include/ecal/protobuf/ecal_proto_message_filter.h %{buildroot}%{_includedir}/ecal/protobuf/ecal_proto_message_filter.h
install -D  _build/include/ecal/protobuf/ecal_proto_maximum_array_dimensions.h %{buildroot}%{_includedir}/ecal/protobuf/ecal_proto_maximum_array_dimensions.h
install -D  _build/include/ecal/protobuf/ecal_proto_hlp.h %{buildroot}%{_includedir}/ecal/protobuf/ecal_proto_hlp.h
install -D  _build/include/ecal/protobuf/ecal_proto_dyn.h %{buildroot}%{_includedir}/ecal/protobuf/ecal_proto_dyn.h
install -D  _build/include/ecal/protobuf/ecal_proto_decoder.h %{buildroot}%{_includedir}/ecal/protobuf/ecal_proto_decoder.h
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/target_definitions/targets_protobuf.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/target_definitions/targets_protobuf.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_python.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_python.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_cpp.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_cpp.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_files.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_files.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/msvc_helper/msvc_macros.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/msvc_helper/msvc_macros.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/git/git_revision_information.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/git/git_revision_information.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/cmake_functions.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/cmake_functions.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/CMakeFunctionsConfigVersion.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/CMakeFunctionsConfigVersion.cmake
install -D  _build/lib/cmake/CMakeFunctions-0.4.1/CMakeFunctionsConfig.cmake %{buildroot}%{_libdir}/cmake/CMakeFunctions-0.4.1/CMakeFunctionsConfig.cmake
install -D  _build/lib64/libecal_hdf5.a %{buildroot}%{_libdir}/libecal_hdf5.a
install -D  _build/lib64/libecal_rec_addon_core.a %{buildroot}%{_libdir}/libecal_rec_addon_core.a
install -D  _build/lib64/libecal_mon_plugin_lib.a %{buildroot}%{_libdir}/libecal_mon_plugin_lib.a
install -D  _build/lib64/libecal_ecal-utils.a %{buildroot}%{_libdir}/libecal_ecal-utils.a
install -D  _build/lib64/libecal_core.so.0.0.0 %{buildroot}%{_libdir}/libecal_core.so.0.0.0
install -D  _build/lib64/libecal_core_c.so.0.0.0 %{buildroot}%{_libdir}/libecal_core_c.so.0.0.0
install -D  _build/lib64/libecal_app_pb.so.0.0.0 %{buildroot}%{_libdir}/libecal_app_pb.so.0.0.0
install -D  _build/lib64/libecal_core_pb.so.0.0.0 %{buildroot}%{_libdir}/libecal_core_pb.so.0.0.0
install -D _build/lib64/libecal_app_pb.so.0 %{buildroot}%{_libdir}/libecal_app_pb.so.0
install -D _build/lib64/libecal_core.so.0 %{buildroot}%{_libdir}/libecal_core.so.0
install -D _build/lib64/libecal_core_c.so.0 %{buildroot}%{_libdir}/libecal_core_c.so.0
install -D _build/lib64/libecal_core_pb.so.0 %{buildroot}%{_libdir}/libecal_core_pb.so.0
install -D  _build/lib64/libecal_pb.a %{buildroot}%{_libdir}/libecal_pb.a
install -D  _build/lib64/libecal_proto.a %{buildroot}%{_libdir}/libecal_proto.a
install -D  _build/lib64/cmake/eCAL/eCALTargets-noconfig.cmake %{buildroot}%{_libdir}/cmake/eCAL/eCALTargets-noconfig.cmake
install -D  _build/lib64/cmake/eCAL/eCALTargets.cmake %{buildroot}%{_libdir}/cmake/eCAL/eCALTargets.cmake
install -D  _build/lib64/cmake/eCAL/helper_functions/ecal_install_functions.cmake %{buildroot}%{_libdir}/cmake/eCAL/helper_functions/ecal_install_functions.cmake
install -D  _build/lib64/cmake/eCAL/helper_functions/ecal_helper_functions.cmake %{buildroot}%{_libdir}/cmake/eCAL/helper_functions/ecal_helper_functions.cmake
install -D  _build/lib64/cmake/eCAL/helper_functions/ecal_add_functions.cmake %{buildroot}%{_libdir}/cmake/eCAL/helper_functions/ecal_add_functions.cmake
install -D  _build/lib64/cmake/eCAL/eCALConfigVersion.cmake %{buildroot}%{_libdir}/cmake/eCAL/eCALConfigVersion.cmake
install -D  _build/lib64/cmake/eCAL/eCALConfig.cmake %{buildroot}%{_libdir}/cmake/eCAL/eCALConfig.cmake

%files
%{_sysconfdir}/ecal/ecal.ini
%{_bindir}/ecal_process_stub-0.0.0
%{_includedir}/ecalhdf5/eh5_defs.h
%{_includedir}/ecalhdf5/eh5_meas.h
%{_includedir}/ecalhdf5/eh5_types.h
%{_includedir}/ecal/measurement/imeasurement.h
%{_includedir}/ecal/measurement/measurement.h
%{_includedir}/ecal/measurement/omeasurement.h
%{_includedir}/ecal/rec/concurrent_status_interface.h
%{_includedir}/ecal/rec/recorder_impl_base.h
%{_includedir}/ecal/rec/recorder_impl_base_types.h
%{_includedir}/ecal/mon/plugin.h
%{_includedir}/ecal/mon/plugin_interface.h
%{_includedir}/ecal/mon/plugin_widget_interface.h
%{_includedir}/ecal/cimpl/ecal_callback_cimpl.h
%{_includedir}/ecal/cimpl/ecal_client_cimpl.h
%{_includedir}/ecal/cimpl/ecal_core_cimpl.h
%{_includedir}/ecal/cimpl/ecal_event_cimpl.h
%{_includedir}/ecal/cimpl/ecal_init_cimpl.h
%{_includedir}/ecal/cimpl/ecal_log_cimpl.h
%{_includedir}/ecal/cimpl/ecal_monitoring_cimpl.h
%{_includedir}/ecal/cimpl/ecal_process_cimpl.h
%{_includedir}/ecal/cimpl/ecal_proto_dyn_json_subscriber_cimpl.h
%{_includedir}/ecal/cimpl/ecal_publisher_cimpl.h
%{_includedir}/ecal/cimpl/ecal_qos_cimpl.h
%{_includedir}/ecal/cimpl/ecal_server_cimpl.h
%{_includedir}/ecal/cimpl/ecal_service_cimpl.h
%{_includedir}/ecal/cimpl/ecal_service_info_cimpl.h
%{_includedir}/ecal/cimpl/ecal_subscriber_cimpl.h
%{_includedir}/ecal/cimpl/ecal_time_cimpl.h
%{_includedir}/ecal/cimpl/ecal_timer_cimpl.h
%{_includedir}/ecal/cimpl/ecal_tlayer_cimpl.h
%{_includedir}/ecal/cimpl/ecal_util_cimpl.h
%{_includedir}/ecal/ecal.h
%{_includedir}/ecal/ecal_callback.h
%{_includedir}/ecal/ecal_clang.h
%{_includedir}/ecal/ecal_client.h
%{_includedir}/ecal/ecal_config.h
%{_includedir}/ecal/ecal_core.h
%{_includedir}/ecal/ecal_event.h
%{_includedir}/ecal/ecal_eventhandle.h
%{_includedir}/ecal/ecal_init.h
%{_includedir}/ecal/ecal_log.h
%{_includedir}/ecal/ecal_log_level.h
%{_includedir}/ecal/ecal_monitoring.h
%{_includedir}/ecal/ecal_os.h
%{_includedir}/ecal/ecal_process.h
%{_includedir}/ecal/ecal_process_mode.h
%{_includedir}/ecal/ecal_process_severity.h
%{_includedir}/ecal/ecal_publisher.h
%{_includedir}/ecal/ecal_qos.h
%{_includedir}/ecal/ecal_server.h
%{_includedir}/ecal/ecal_service.h
%{_includedir}/ecal/ecal_service_info.h
%{_includedir}/ecal/ecal_subscriber.h
%{_includedir}/ecal/ecal_time.h
%{_includedir}/ecal/ecal_timed_cb.h
%{_includedir}/ecal/ecal_timer.h
%{_includedir}/ecal/ecal_tlayer.h
%{_includedir}/ecal/ecal_util.h
%{_includedir}/ecal/ecalc.h
%{_includedir}/ecal/ecalc_types.h
%{_includedir}/ecal/msg/message.h
%{_includedir}/ecal/msg/proto/message.h
%{_includedir}/ecal/msg/capnproto/dynamic.h
%{_includedir}/ecal/msg/capnproto/helper.h
%{_includedir}/ecal/msg/capnproto/publisher.h
%{_includedir}/ecal/msg/capnproto/subscriber.h
%{_includedir}/ecal/msg/dynamic.h
%{_includedir}/ecal/msg/flatbuffers/publisher.h
%{_includedir}/ecal/msg/flatbuffers/subscriber.h
%{_includedir}/ecal/msg/messagepack/publisher.h
%{_includedir}/ecal/msg/messagepack/subscriber.h
%{_includedir}/ecal/msg/protobuf/client.h
%{_includedir}/ecal/msg/protobuf/dynamic_json_subscriber.h
%{_includedir}/ecal/msg/protobuf/dynamic_publisher.h
%{_includedir}/ecal/msg/protobuf/dynamic_subscriber.h
%{_includedir}/ecal/msg/protobuf/publisher.h
%{_includedir}/ecal/msg/protobuf/server.h
%{_includedir}/ecal/msg/protobuf/subscriber.h
%{_includedir}/ecal/msg/publisher.h
%{_includedir}/ecal/msg/string/message.h
%{_includedir}/ecal/msg/string/publisher.h
%{_includedir}/ecal/msg/string/subscriber.h
%{_includedir}/ecal/msg/subscriber.h
%{_includedir}/ecal/ecal_defs.h
%{_includedir}/ecal/app/pb/sys/state.pb.h
%{_includedir}/ecal/app/pb/sys/client_service.pb.h
%{_includedir}/ecal/app/pb/sys/process.pb.h
%{_includedir}/ecal/app/pb/sys/service.pb.h
%{_includedir}/ecal/app/pb/mma/mma.pb.h
%{_includedir}/ecal/app/pb/play/service.pb.h
%{_includedir}/ecal/app/pb/play/state.pb.h
%{_includedir}/ecal/app/pb/rec/client_service.pb.h
%{_includedir}/ecal/app/pb/rec/client_state.pb.h
%{_includedir}/ecal/app/pb/rec/server_config.pb.h
%{_includedir}/ecal/app/pb/rec/server_service.pb.h
%{_includedir}/ecal/app/pb/rec/server_state.pb.h
%{_includedir}/ecal/core/pb/topic.pb.h
%{_includedir}/ecal/core/pb/ecal.pb.h
%{_includedir}/ecal/core/pb/host.pb.h
%{_includedir}/ecal/core/pb/layer.pb.h
%{_includedir}/ecal/core/pb/monitoring.pb.h
%{_includedir}/ecal/core/pb/process.pb.h
%{_includedir}/ecal/core/pb/service.pb.h
%{_includedir}/ecal/pb/ecal.pb.h
%{_includedir}/ecal/pb/host.pb.h
%{_includedir}/ecal/pb/layer.pb.h
%{_includedir}/ecal/pb/mma/mma.pb.h
%{_includedir}/ecal/pb/monitoring.pb.h
%{_includedir}/ecal/pb/play/service.pb.h
%{_includedir}/ecal/pb/play/state.pb.h
%{_includedir}/ecal/pb/process.pb.h
%{_includedir}/ecal/pb/rec/client_service.pb.h
%{_includedir}/ecal/pb/rec/client_state.pb.h
%{_includedir}/ecal/pb/rec/server_config.pb.h
%{_includedir}/ecal/pb/rec/server_service.pb.h
%{_includedir}/ecal/pb/rec/server_state.pb.h
%{_includedir}/ecal/pb/service.pb.h
%{_includedir}/ecal/pb/sim_time.pb.h
%{_includedir}/ecal/pb/sys/client_service.pb.h
%{_includedir}/ecal/pb/sys/service.pb.h
%{_includedir}/ecal/pb/sys/state.pb.h
%{_includedir}/ecal/pb/topic.pb.h
%{_includedir}/ecal/protobuf/ecal_proto_visitor.h
%{_includedir}/ecal/protobuf/ecal_proto_message_filter.h
%{_includedir}/ecal/protobuf/ecal_proto_maximum_array_dimensions.h
%{_includedir}/ecal/protobuf/ecal_proto_hlp.h
%{_includedir}/ecal/protobuf/ecal_proto_dyn.h
%{_includedir}/ecal/protobuf/ecal_proto_decoder.h
%{_libdir}/libecal_app_pb.so.0
%{_libdir}/libecal_core.so.0
%{_libdir}/libecal_core_c.so.0
%{_libdir}/libecal_core_pb.so.0
%{_libdir}/cmake/CMakeFunctions-0.4.1/target_definitions/targets_protobuf.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_python.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_cpp.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/protoc_functions/protoc_generate_files.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/msvc_helper/msvc_macros.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/git/git_revision_information.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/cmake_functions.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/CMakeFunctionsConfigVersion.cmake
%{_libdir}/cmake/CMakeFunctions-0.4.1/CMakeFunctionsConfig.cmake
%{_libdir}/libecal_hdf5.a
%{_libdir}/libecal_rec_addon_core.a
%{_libdir}/libecal_mon_plugin_lib.a
%{_libdir}/libecal_ecal-utils.a
%{_libdir}/libecal_core.so.0.0.0
%{_libdir}/libecal_core_c.so.0.0.0
%{_libdir}/libecal_app_pb.so.0.0.0
%{_libdir}/libecal_core_pb.so.0.0.0
%{_libdir}/libecal_pb.a
%{_libdir}/libecal_proto.a
%{_libdir}/cmake/eCAL/eCALTargets-noconfig.cmake
%{_libdir}/cmake/eCAL/eCALTargets.cmake
%{_libdir}/cmake/eCAL/helper_functions/ecal_install_functions.cmake
%{_libdir}/cmake/eCAL/helper_functions/ecal_helper_functions.cmake
%{_libdir}/cmake/eCAL/helper_functions/ecal_add_functions.cmake
%{_libdir}/cmake/eCAL/eCALConfigVersion.cmake
%{_libdir}/cmake/eCAL/eCALConfig.cmake
%license LICENSE.txt
%doc README.md

%changelog
* Tue Jan 17 2023 Leonardo Rossetti <lrossett@redhat.com> - 5.11.1-1
- First version being packaged
