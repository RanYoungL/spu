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
      ->regKernel<                                              //
          fantastic4::P2A, fantastic4::V2A, fantastic4::A2P, fantastic4::A2V
          >();
}

std::unique_ptr<SPUContext> makeFantastic4Protocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regFantastic4Protocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
