#include "libspu/mpc/fantastic4/protocol.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/fantastic4/arithmetic.h"
#include "libspu/mpc/fantastic4/boolean.h"
#include "libspu/mpc/fantastic4/conversion.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/standard_shape/protocol.h"

#define ENABLE_PRECISE_ABY3_TRUNCPR

namespace spu::mpc {

void regFantastic4Protocol(SPUContext* ctx,
                           const std::shared_ptr<yacl::link::Context>& lctx) {
  fantastic4::registerTypes();

  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  ctx->prot()
          ->regKernel <  //
                         //   fantastic4::P2A, fantastic4::V2A, fantastic4::A2P,
                         //   fantastic4::A2V,           // Conversions
                         //   fantastic4::B2P, fantastic4::P2B, fantastic4::A2B,
                         //   // Conversion2 fantastic4::B2ASelector,
                         //   /*fantastic4::B2AByOT, fantastic4::B2AByPPA*/  //
                         //   B2A fantastic4::CastTypeB, // Cast
                         //   fantastic4::NegateA, // Negate
      fantastic4::AddAP,
      fantastic4::AddAA,  // Add
      //   fantastic4::MulAP, fantastic4::MulAA, fantastic4::MulA1B, // Mul
      //   fantastic4::MatMulAP, fantastic4::MatMulAA,                       //
      //   MatMul fantastic4::LShiftA, fantastic4::LShiftB, // LShift
      //   fantastic4::RShiftB, fantastic4::ARShiftB,                        //
      //   (A)Rshift fantastic4::MsbA2B, // MSB
      fantastic4::EqualAA, fantastic4::EqualAP,  // Equal
      //   fantastic4::CommonTypeB, fantastic4::CommonTypeV,                 //
      //   CommonType fantastic4::AndBP, fantastic4::AndBB, // And
      fantastic4::XorBP, fantastic4::XorBB,  // Xor
  //   fantastic4::BitrevB,                                        // bitreverse
  //   fantastic4::BitIntlB, fantastic4::BitDeintlB,  // bit(de)interleave
  //          fantastic4::RandA,                       // rand
  // #ifdef ENABLE_PRECISE_ABY3_TRUNCPR
  //           fantastic4::TruncAPr,  // Trunc
  // #else
  //           fantastic4::TruncA,
  // #endif
  //           fantastic4::OramOneHotAA, fantastic4::OramOneHotAP,
  //           fantastic4::OramReadOA,      // oram fantastic4::OramReadOP, //
  //           oram fantastic4::RandPermM, fantastic4::PermAM,
  //           fantastic4::PermAP, fantastic4::InvPermAM,  // perm
  //           fantastic4::InvPermAP // perm
  //           >();
}

std::unique_ptr<SPUContext> makeFantastic4Protocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regFantastic4Protocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
