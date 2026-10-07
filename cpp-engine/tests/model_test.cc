#include <catch2/catch_test_macros.hpp>
#include "model.h"
#include <fstream>
#include <cstdio>
#include <cstdint>
#include <cstring>
static void word(std::ostream& out, uint32_t value) {
  for (int i=0;i<4;++i) out.put(char(value >> (8*i)));
}
TEST_CASE("model loads little-endian weights and preserves batch order") {
  const char* path = "model_test_weights.bin";
  {
    std::ofstream out(path, std::ios::binary);
    out.write("MIN1",4); word(out,2); word(out,2); word(out,2);
    // Identity matrices, zero biases: class follows the largest input.
    for (float value : {1.f,0.f,0.f,1.f,0.f,0.f,1.f,0.f,0.f,1.f,0.f,0.f}) {
      uint32_t bits; std::memcpy(&bits,&value,4); word(out,bits);
    }
  }
  Model model(path);
  auto result = model.predict({1,0,0,1},2);
  REQUIRE(result.size()==2);
  CHECK(result[0].predicted_class==0);
  CHECK(result[1].predicted_class==1);
  CHECK(result[0].confidence > .73f);
  CHECK(result[0].confidence < .74f);
  CHECK_THROWS(model.predict({1},1));
  CHECK_THROWS(model.predict({-1,0},1));
  {
    std::ofstream out(path, std::ios::binary | std::ios::app); out.put('x');
  }
  CHECK_THROWS(Model(path));
  std::remove(path);
  CHECK_THROWS(Model(path));
}
