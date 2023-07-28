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

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t  = Kokkos::View<double***, RemoteSpace_t>;
using HostView_t    = typename RemoteView_t::HostMirror;
using UnmanagedView_t =
    Kokkos::View<double***, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

struct CommHelper {
  MPI_Comm comm;
  // Num MPI ranks in each dimension
  int nx, ny, nz;

  // this process
  int me;
  int nranks;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    nx = me;
    ny = 0;
    nz = 0;

    printf("NumRanks: %i Me: %i\n", nranks, me);
  }
};

struct System {
  // Using theoretical physicists' way of describing the system,
  // i.e. we stick to everything in as few constants as possible
  // let i and i+1 two timesteps dt apart:
  // T(x,y,z)_(i+1) =  T(x,y,z)_(i) + dT(x,y,z)*dt
  // dT(x,y,z) = q * sum_{dxdydz}( T(x + dx, y + dy, z + dz) - T(x,y,z))
  // If it's the surface of the body, add
  // dT(x,y,z) += -sigma * T(x,y,z)^4
  // If it's the z == 0 surface, add incoming radiation energy
  // dT(x,y,0) += P

  // Communicator
  CommHelper comm;

  // size of system
  int X, Y, Z;

  // Local box
  int my_lo_x, my_hi_x;

  // number of timesteps
  int N;

  // interval for print
  int I;

  // Temperature and delta Temperature
  RemoteView_t T, dT;
  UnmanagedView_t dT_u;
  HostView_t T_h;

  // Initial Temmperature
  double T0;

  // timestep width
  double dt;

  // thermal transfer coefficient
  double q;

  // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double sigma;

  // incoming power
  double P;

  // init_system
  System(MPI_Comm comm_) : comm(comm_) {
    // populate with defaults, set the rest in setup_subdomain.
    X = Y = Z = 200;
    my_lo_x            = 0;
    my_hi_x            = 0;
    N                  = 1000;
    I                  = 100;
    T_h                = HostView_t();
    T                  = RemoteView_t();
    dT                 = RemoteView_t();
    dT_u               = UnmanagedView_t();
    T0                 = 0.0;
    dt                 = 0.1;
    q                  = 1.0;
    sigma              = 1.0;
    P                  = 1.0;
  }
  void destroy_exec_spaces() {
    ;
  }

  void setup_subdomain() {
    auto local_range = Kokkos::Experimental::get_local_range(X);
    my_lo_x          = local_range.first;
    my_hi_x          = local_range.second + 1;

    printf("My Domain: %i (%i %i %i) (%i %i %i)\n", comm.me, my_lo_x, 0,
           0, my_hi_x, Y, Z);
    T   = RemoteView_t("System::T", X, Y, Z);
    T_h = HostView_t("Host::T", T.extent(0), Y, Z);
    dT  = RemoteView_t("System::dT", X, Y, Z);
    dT_u = UnmanagedView_t(dT.data(), X, Y, Z);

    Kokkos::deep_copy(T_h, T0);
    Kokkos::deep_copy(T, T_h);
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -X IARG: (%i) num elements in the X direction\n", X);
    printf("  -Y IARG: (%i) num elements in the Y direction\n", Y);
    printf("  -Z IARG: (%i) num elements in the Z direction\n", Z);
    printf("  -N IARG: (%i) num timesteps\n", N);
    printf("  -I IARG: (%i) print interval\n", I);
    printf("  -T0 FARG: (%lf) initial temperature\n", T0);
    printf("  -dt FARG: (%lf) timestep size\n", dt);
    printf("  -q FARG: (%lf) thermal conductivity\n", q);
    printf("  -sigma FARG: (%lf) thermal radiation\n", sigma);
    printf("  -P FARG: (%lf) incoming power\n", P);
  }

  // check command line args
  bool check_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        print_help();
        return false;
      }
    }
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-X") == 0) X = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Y") == 0) Y = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Z") == 0) Z = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-I") == 0) I = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-T0") == 0) T0 = atof(argv[i + 1]);
      if (strcmp(argv[i], "-dt") == 0) dt = atof(argv[i + 1]);
      if (strcmp(argv[i], "-q") == 0) q = atof(argv[i + 1]);
      if (strcmp(argv[i], "-sigma") == 0) sigma = atof(argv[i + 1]);
      if (strcmp(argv[i], "-P") == 0) P = atof(argv[i + 1]);
    }
    setup_subdomain();
    return true;
  }

  // compute both inner and outer updates. This function is suitable for both.
  struct ComputeAllDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeAllDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    // printf("begin    computeAllDT with x,y,z=(%i,%i,%i)\n", x, y, z);
    int num_surfaces = 0;
    if (x > 0) {
      dT_xyz += q * (T(x - 1, y, z) - T_xyz);
      // printf("x access computeAllDT with x,y,z=(%i,%i,%i)\n", x, y, z);
    } else {
      num_surfaces += 1;
      // Incoming Power
      if (x == 0) dT_xyz += P;
    }

    if (x < X - 1) {
      dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    } else {
      num_surfaces += 1;
    }

    if (y > 0) {
      dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    } else {
      num_surfaces += 1;
    }

    if (y < Y - 1) {
      dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    } else {
      num_surfaces += 1;
    }

    if (z > 0) {
      dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    } else {
      num_surfaces += 1;
    }

    if (z < Z - 1) {
      dT_xyz += q * (T(x, y, z + 1) - T_xyz);
    } else {
      num_surfaces += 1;
    }

    // radiation
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;

    dT_u(x, y, z) = dT_xyz;
  }

  void compute_all_dT() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeAllDT, int>;
    Kokkos::parallel_for(
        "ComputeAllDT",
        policy_t({my_lo_x, 0, 0}, {my_hi_x, Y, Z}, {16, 8, 8}),
        *this);
  }

  // Some compilers have deduction issues if this were just a tagget operator
  // So it is instead a full Functor
  struct computeT {
    RemoteView_t T;
    UnmanagedView_t dT_u;
    double dt;
    computeT(RemoteView_t T_, UnmanagedView_t dT_u_, double dt_)
        : T(T_), dT_u(dT_u_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double& sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT_u(x, y, z);
    }
  };

  double compute_T() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    double my_T;
    Kokkos::parallel_reduce(
        "ComputeT",
        policy_t({my_lo_x, 0, 0}, {my_hi_x, Y, Z}, {10, 10, 10}),
        computeT(T, dT_u, dt), my_T);
    double sum_T;
    RemoteSpace_t().fence();
    Kokkos::DefaultExecutionSpace().fence();
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM,
                  comm.comm); /* also a barrier */
    return sum_T;
  }

  // run time loops
  void timestep() {
    Kokkos::Timer timer;
    double old_time = 0.0;
    double time_a, time_c, time_d;
    double time_compute, time_all;
    time_all = time_compute = 0.0;
    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0; /* stop heat in halfway through */
      time_a = timer.seconds();
      compute_all_dT();
      RemoteSpace_t().fence();
      Kokkos::DefaultExecutionSpace().fence();
      time_c       = timer.seconds();
      double T_ave = compute_T();
      time_d       = timer.seconds();
      time_all += time_c - time_a;
      time_compute += time_d - time_c;
      T_ave /= 1e-9 * (X * Y * Z);
      if ((t % I == 0 || t == N) && (comm.me == 0)) {
        double time = timer.seconds();
        printf("%d T=%lf Time (%lf %lf)\n", t, T_ave, time, time - old_time);
        printf("     inner + surface: %lf compute: %lf\n", time_all,
               time_compute);
        old_time = time;
      }
    }
  }
};

int main(int argc, char* argv[]) {
  int mpi_thread_level_available;
  int mpi_thread_level_required = MPI_THREAD_MULTIPLE;

#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
  mpi_thread_level_required = MPI_THREAD_SINGLE;
#endif

  MPI_Init_thread(&argc, &argv, mpi_thread_level_required,
                  &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);

#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

#ifdef KRS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  Kokkos::initialize(argc, argv);
  {
    System sys(MPI_COMM_WORLD);

    if (sys.check_args(argc, argv)) sys.timestep();
    sys.destroy_exec_spaces();
  }
  Kokkos::finalize();
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#endif
#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
  MPI_Finalize();
  return 0;
}

