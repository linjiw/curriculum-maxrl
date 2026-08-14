// Persistent client for the BARN exact-step Gazebo world plugin.

#include <gazebo/gazebo_client.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>

namespace {
std::mutex receipt_mutex;
std::condition_variable receipt_changed;
int completed_iterations = 0;

void OnStepDone(ConstIntPtr &message) {
  {
    std::lock_guard<std::mutex> lock(receipt_mutex);
    completed_iterations = message->data();
  }
  receipt_changed.notify_all();
}
}  // namespace

int main(int argc, char **argv) {
  gazebo::client::setup(argc, argv);
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();
  auto subscriber = node->Subscribe("~/barn_step_done", OnStepDone);
  auto publisher = node->Advertise<gazebo::msgs::Int>("~/barn_step");
  publisher->WaitForConnection();

  unsigned int iterations = 0;
  while (std::cin >> iterations) {
    if (iterations == 0) {
      std::cout << "error 0" << std::endl;
      continue;
    }
    gazebo::msgs::Int request;
    request.set_data(static_cast<int>(iterations));
    {
      std::lock_guard<std::mutex> lock(receipt_mutex);
      completed_iterations = 0;
    }
    publisher->Publish(request);
    {
      std::unique_lock<std::mutex> lock(receipt_mutex);
      if (!receipt_changed.wait_for(
              lock, std::chrono::seconds(10), [iterations] {
                return completed_iterations == static_cast<int>(iterations);
              })) {
        std::cout << "error timeout " << iterations << std::endl;
        continue;
      }
    }
    std::cout << "ok " << iterations << std::endl;
  }
  gazebo::client::shutdown();
  return 0;
}
