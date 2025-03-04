//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contact: Jan Ciesko (jciesko@sandia.gov)
//
//@HEADER

#ifndef KOKKOS_REMOTESPACES_SHMEM_DATAHANDLE_HPP
#define KOKKOS_REMOTESPACES_SHMEM_DATAHANDLE_HPP

namespace Kokkos {
namespace Impl {

template <class T, class Traits>
struct SHMEMDataHandle {
  T *ptr;

  KOKKOS_INLINE_FUNCTION
  SHMEMDataHandle() : ptr(NULL) {}

  KOKKOS_INLINE_FUNCTION
  SHMEMDataHandle(T *ptr_) : ptr(ptr_) {}

  KOKKOS_INLINE_FUNCTION
  SHMEMDataHandle(SHMEMDataHandle<T, Traits> const &arg) : ptr(arg.ptr) {}

  template <typename SrcTraits>
  KOKKOS_INLINE_FUNCTION SHMEMDataHandle(SrcTraits const &arg) : ptr(arg.ptr) {}

  template <typename iType>
  KOKKOS_INLINE_FUNCTION SHMEMDataElement<T, Traits> operator()(
      const int &pe, const iType &i) const {
    SHMEMDataElement<T, Traits> element(ptr, pe, i);
    return element;
  }

  KOKKOS_INLINE_FUNCTION
  T *operator+(size_t &offset) const { return ptr + offset; }
};

template <class Traits>
struct ViewDataHandle<
    Traits, typename std::enable_if<std::is_same<
                typename Traits::specialize,
                Kokkos::Experimental::RemoteSpaceSpecializeTag>::value>::type> {
  using value_type  = typename Traits::value_type;
  using handle_type = SHMEMDataHandle<value_type, Traits>;
  using return_type = SHMEMDataElement<value_type, Traits>;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const &arg_data_ptr, track_type const & /*arg_tracker*/) {
    return handle_type(arg_data_ptr);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type *arg_data_ptr,
                            track_type const & /*arg_tracker*/) {
    return handle_type(arg_data_ptr);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr + offset);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const arg_data_ptr) {
    return handle_type(arg_data_ptr);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION handle_type operator=(SrcHandleType const &rhs) {
    return handle_type(rhs);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_SHMEM_DATAHANDLE_HPP