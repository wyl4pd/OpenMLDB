//
// file_appender_test.cc
// Copyright (C) 2017 4paradigm.com
// Author vagrant
// Date 2017-04-21
//

#include "replica/log_replicator.h"

#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <stdio.h>
#include "proto/tablet.pb.h"
#include "logging.h"
#include "thread_pool.h"
#include <sofa/pbrpc/pbrpc.h>

#include "timer.h"

using ::baidu::common::ThreadPool;
using ::google::protobuf::RpcController;
using ::google::protobuf::Closure;
using ::baidu::common::INFO;
using ::baidu::common::DEBUG;

namespace rtidb {
namespace replica {

class MockTabletImpl : public ::rtidb::api::TabletServer {

public:
    MockTabletImpl(const ReplicatorRole& role,
                   const std::string& path,
                   const std::vector<std::string>& endpoints): role_(role),
    path_(path), endpoints_(endpoints), replicator_(path_, endpoints_, role_, 1, 1){
    }

    MockTabletImpl(const ReplicatorRole& role,
                   const std::string& path,
                   ApplyLogFunc func): role_(role),
    path_(path), func_(func), replicator_(path_, func_, role_, 1, 1){
    }
    ~MockTabletImpl() {
        replicator_.Stop();
    }

    bool Init() {
        return replicator_.Init();
    }

    void Put(RpcController* controller,
             const ::rtidb::api::PutRequest* request,
             ::rtidb::api::PutResponse* response,
             Closure* done) {}

    void Scan(RpcController* controller,
              const ::rtidb::api::ScanRequest* request,
              ::rtidb::api::ScanResponse* response,
              Closure* done) {}

    void CreateTable(RpcController* controller,
            const ::rtidb::api::CreateTableRequest* request,
            ::rtidb::api::CreateTableResponse* response,
            Closure* done) {}

    void DropTable(RpcController* controller,
            const ::rtidb::api::DropTableRequest* request,
            ::rtidb::api::DropTableResponse* response,
            Closure* done) {}

    void AppendEntries(RpcController* controller,
            const ::rtidb::api::AppendEntriesRequest* request,
            ::rtidb::api::AppendEntriesResponse* response,
            Closure* done) {} 
private:
    ReplicatorRole role_;
    std::string path_;
    std::vector<std::string> endpoints_;
    ApplyLogFunc func_;
    LogReplicator replicator_;

};

bool ReceiveEntry(const ::rtidb::api::LogEntry& entry) {
    return true;
}

class LogReplicatorTest : public ::testing::Test {

public:
    LogReplicatorTest() {}

    ~LogReplicatorTest() {}
};

inline std::string GenRand() {
    return boost::lexical_cast<std::string>(rand() % 10000000 + 1);
}

bool StartRpcServe(MockTabletImpl* tablet,
        const std::string& endpoint) {
    sofa::pbrpc::RpcServerOptions options;
    sofa::pbrpc::RpcServer rpc_server(options);
    if (!rpc_server.RegisterService(tablet)) {
        return false;
    }
    bool ok =rpc_server.Start(endpoint);
    if (ok) {
        LOG(INFO, "register service ok");
    }else {
        LOG(WARNING, "fail to start service");
    }
    return ok;
}

TEST_F(LogReplicatorTest, Init) {
    std::vector<std::string> endpoints;
    std::string folder = "/tmp/rtidb/" + GenRand() + "/";
    LogReplicator replicator(folder, endpoints, kLeaderNode, 1, 1);
    bool ok = replicator.Init();
    ASSERT_TRUE(ok);
    replicator.Stop();
}

TEST_F(LogReplicatorTest, BenchMark) {
    std::vector<std::string> endpoints;
    std::string folder = "/tmp/rtidb/" + GenRand() + "/";
    LogReplicator replicator(folder, endpoints, kLeaderNode, 1, 1);
    bool ok = replicator.Init();
    ::rtidb::api::LogEntry entry;
    entry.set_term(1);
    entry.set_pk("test");
    entry.set_value("test");
    entry.set_ts(9527);
    ok = replicator.AppendEntry(entry);
    ASSERT_TRUE(ok);
    replicator.Stop();
}

bool MockApplyLog(const ::rtidb::api::LogEntry& entry) {
    LOG(INFO, "apply entry pk %s, value %s , ts %lld", entry.pk().c_str(), entry.value().c_str(), entry.ts());
    return true;
}

TEST_F(LogReplicatorTest, LeaderAndFollower) {
    sofa::pbrpc::RpcServerOptions options;
    sofa::pbrpc::RpcServer rpc_server0(options);
    sofa::pbrpc::RpcServer rpc_server1(options);
    {

        std::string follower_addr = "127.0.0.1:18527";
        std::string folder = "/tmp/rtidb/" + GenRand() + "/";
        MockTabletImpl* follower = new MockTabletImpl(kFollowerNode, folder, boost::bind(MockApplyLog, _1)); 
        bool ok = follower->Init();
        ASSERT_TRUE(ok);
       if (!rpc_server0.RegisterService(follower)) {
            ASSERT_TRUE(false);
        }
        ok =rpc_server0.Start(follower_addr);
        ASSERT_TRUE(ok);
        LOG(INFO, "start follower");

    }
    {

        std::string follower_addr = "127.0.0.1:18528";
        std::string folder = "/tmp/rtidb/" + GenRand() + "/";
        MockTabletImpl* follower = new MockTabletImpl(kFollowerNode, folder, boost::bind(MockApplyLog, _1)); 
        bool ok = follower->Init();
        ASSERT_TRUE(ok);
        if (!rpc_server1.RegisterService(follower)) {
            ASSERT_TRUE(false);
        }
        ok =rpc_server1.Start(follower_addr);
        ASSERT_TRUE(ok);
        LOG(INFO, "start follower");
    }

    std::vector<std::string> endpoints;
    endpoints.push_back("127.0.0.1:18527");
    std::string folder = "/tmp/rtidb/" + GenRand() + "/";
    LogReplicator leader(folder, endpoints, kLeaderNode, 1, 1);
    bool ok = leader.Init();
    ASSERT_TRUE(ok);
    ::rtidb::api::LogEntry entry;
    entry.set_pk("test_pk");
    entry.set_value("value1");
    entry.set_ts(9527);
    ok = leader.AppendEntry(entry);
    entry.set_value("value2");
    ok = leader.AppendEntry(entry);
    entry.set_value("value3");
    ok = leader.AppendEntry(entry);
    entry.set_value("value4");
    ok = leader.AppendEntry(entry);
    entry.set_value("value5");
    ok = leader.AppendEntry(entry);
    entry.set_value("value6");
    ok = leader.AppendEntry(entry);
    leader.Notify();
    leader.AddReplicateNode("127.0.0.1:18528");
    sleep(4);
    leader.Stop();
    ASSERT_TRUE(ok);
}

}
}

int main(int argc, char** argv) {
    srand (time(NULL));
    ::baidu::common::SetLogLevel(::baidu::common::DEBUG);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

