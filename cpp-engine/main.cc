#include "model.h"
#include "generated/inference.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <iostream>
class Engine final : public inference::InferenceEngine::Service {
 public:
  explicit Engine(const Model& model) : model_(model) {}
  grpc::Status PredictBatch(grpc::ServerContext* context,
      const inference::PredictBatchRequest* request, inference::PredictBatchResponse* response) override {
    if (context->IsCancelled()) return {grpc::StatusCode::CANCELLED, "request cancelled"};
    if (request->image_size() < 1 || request->image_size() > 8)
      return {grpc::StatusCode::INVALID_ARGUMENT, "batch must contain 1..8 images"};
    std::vector<float> pixels;
    for (const auto& image : request->image()) {
      if (image.pixels_size() != model_.input_dim())
        return {grpc::StatusCode::INVALID_ARGUMENT, "wrong input dimension"};
      pixels.insert(pixels.end(), image.pixels().begin(), image.pixels().end());
    }
    try {
      for (auto prediction : model_.predict(pixels, request->image_size())) {
        auto* result = response->add_predictions();
        result->set_predicted_class(prediction.predicted_class);
        result->set_confidence(prediction.confidence);
      }
      return grpc::Status::OK;
    } catch (const std::invalid_argument& error) {
      return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
    } catch (const std::exception& error) {
      return {grpc::StatusCode::INTERNAL, error.what()};
    }
  }
 private:
  const Model& model_;
};
int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: inference_engine weights.bin [address]\n";
    return 1;
  }
  try {
    Model model(argv[1]);
    if (model.input_dim() != 784) throw std::runtime_error("serving requires 784 inputs");
    Engine service(model);
    grpc::ServerBuilder builder;
    std::string address = "127.0.0.1:50051";
    if (argc == 3) {
      address = argv[2];
    }
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    if (!server) throw std::runtime_error("could not bind gRPC server");
    std::cout << "engine listening on " << address << std::endl;
    server->Wait();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
