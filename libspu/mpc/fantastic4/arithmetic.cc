#include "libspu/mpc/fantastic4/arithmetic.h"

#include <future>


#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"


namespace spu::mpc::fantastic4 {

// ///////////////////////////////////////////////////
// Layout of Rep4:
// P1(x1,x2,x3) P2(x2,x3,x4) P3(x3,x4,x1) P4(x4,x1,x2)
// ///////////////////////////////////////////////////


// Pass the third share to previous party
NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto numel = in.numel();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<Pub2kTy>(field), in.shape());
    NdArrayView<pshr_el_t> _out(out);
    NdArrayView<ashr_t> _in(in);

    std::vector<ashr_el_t> x3(numel);

    pforeach(0, numel, [&](int64_t idx) { x3[idx] = _in[idx][2]; });

    auto x4 = comm->rotate<ashr_el_t>(x3, "a2p");  // comm => 1, k

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + _in[idx][2] + x4[idx];
    });

    return out;
  });
}

// x1 = x, x2 = x3 = x4 = 0

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;


    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<pshr_el_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = rank == 0 ? _in[idx] : 0;
      _out[idx][1] = rank == 3 ? _in[idx] : 0;
      _out[idx][2] = rank == 2 ? _in[idx] : 0;
    });

    // TODO: debug masks?

    return out;
  });
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using vshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_t> _in(in);
    auto out_ty = makeType<Priv2kTy>(field, rank);

    if (comm->getRank() == rank) {
      auto x4 = comm->recv<ashr_el_t>(comm->nextRank(), "a2v");  // comm => 1, k
                                                                 //
      NdArrayRef out(out_ty, in.shape());
      NdArrayView<vshr_el_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = _in[idx][0] + _in[idx][1] + _in[idx][2] + x4[idx];
      });
      return out;

    } else if (comm->getRank() == (rank + 1) % 4) {
      std::vector<ashr_el_t> x3(in.numel());

      pforeach(0, in.numel(), [&](int64_t idx) { x3[idx] = _in[idx][2]; });

      comm->sendAsync<ashr_el_t>(comm->prevRank(), x3,
                                 "a2v");  // comm => 1, k
      return makeConstantArrayRef(out_ty, in.shape());
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}



// /////////////////////////////////////////////////
// V2A
// In aby3, no use of prg, the dealer just distribute shr1 and shr2, set shr3 = 0
// /////////////////////////////////////////////////
NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const auto field = in_ty->field();

  size_t owner_rank = in_ty->owner();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);

    if (comm->getRank() == owner_rank) {
      auto splits = ring_rand_additive_splits(in, 3);
      // send (shr2, shr3) to next party
      //      (shr3, shr1) to next next party
      //      (shr1, shr2) to prev party
      // shr4 = 0

      comm->sendAsync((owner_rank + 1) % 4, splits[1], "v2a 1");  // comm => 1, k
      comm->sendAsync((owner_rank + 1) % 4, splits[2], "v2a 2");  // comm => 1, k

      comm->sendAsync((owner_rank + 2) % 4, splits[2], "v2a 1");  // comm => 1, k
      comm->sendAsync((owner_rank + 2) % 4, splits[0], "v2a 2");  // comm => 1, k

      comm->sendAsync((owner_rank + 3) % 4, splits[0], "v2a 1");  // comm => 1, k
      comm->sendAsync((owner_rank + 3) % 4, splits[1], "v2a 2");  // comm => 1, k


      NdArrayView<ashr_el_t> _s0(splits[0]);
      NdArrayView<ashr_el_t> _s1(splits[1]);
      NdArrayView<ashr_el_t> _s2(splits[2]);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = _s0[idx];
        _out[idx][1] = _s1[idx];
        _out[idx][1] = _s2[idx];
      });
    } 
    else if (comm->getRank() == (owner_rank + 1) % 4) {
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 3) % 4, "v2a 1");  // comm => 1, k
      auto x2 = comm->recv<ashr_el_t>((comm->getRank() + 3) % 4, "v2a 2");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        
        _out[idx][0] = x1[idx];
        _out[idx][1] = x2[idx];
        _out[idx][2] = 0;
      });
    } 
    else if (comm->getRank() == (owner_rank + 2) % 4) {
      auto x3 = comm->recv<ashr_el_t>((comm->getRank() + 2) % 4, "v2a 1");  // comm => 1, k
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 2) % 4, "v2a 2");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = x3[idx];
        _out[idx][1] = 0;
        _out[idx][2] = x1[idx];
      });
    } else {
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 1) % 4, "v2a 1");  // comm => 1, k
      auto x2 = comm->recv<ashr_el_t>((comm->getRank() + 1) % 4, "v2a 2");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = x1[idx];
        _out[idx][2] = x2[idx];
      });
    }

    return out;
  });
}




NdArrayRef NegateA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = std::make_unsigned_t<ring2k_t>;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = -_in[idx][0];
      _out[idx][1] = -_in[idx][1];
      _out[idx][2] = -_in[idx][2];
    });

    return out;
  });
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      _out[idx][2] = _lhs[idx][2];
      if (rank == 0) _out[idx][2] += _rhs[idx];
      if (rank == 1) _out[idx][1] += _rhs[idx];
      if (rank == 2) _out[idx][0] += _rhs[idx];
    });
    return out;
  });
}

NdArrayRef AddAA::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using shr_t = std::array<ring2k_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] + _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] + _rhs[idx][1];
      _out[idx][2] = _lhs[idx][2] + _rhs[idx][2];
    });
    return out;
  });
}


}





