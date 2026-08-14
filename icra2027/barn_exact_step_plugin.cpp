// Gazebo Classic world plugin for exact, acknowledged CPU physics stepping.

#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

namespace gazebo {
class BarnExactStepPlugin : public WorldPlugin {
 public:
  void Load(physics::WorldPtr world, sdf::ElementPtr) override {
    world_ = std::move(world);
    node_.reset(new transport::Node());
    node_->Init(world_->Name());
    done_publisher_ = node_->Advertise<msgs::Int>("~/barn_step_done");
    step_subscriber_ = node_->Subscribe(
        "~/barn_step", &BarnExactStepPlugin::OnStep, this);
  }

 private:
  void OnStep(ConstIntPtr &message) {
    const int iterations = message->data();
    if (iterations <= 0) {
      return;
    }
    world_->SetPaused(true);
    world_->Step(static_cast<unsigned int>(iterations));
    msgs::Int done;
    done.set_data(iterations);
    done_publisher_->Publish(done);
  }

  physics::WorldPtr world_;
  transport::NodePtr node_;
  transport::PublisherPtr done_publisher_;
  transport::SubscriberPtr step_subscriber_;
};

GZ_REGISTER_WORLD_PLUGIN(BarnExactStepPlugin)
}  // namespace gazebo
