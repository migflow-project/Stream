#include "solver.hpp"
#include "linear_system.hpp"
#include "ava_device_array.hpp"
#include "ava_host_array.hpp"
#include "ava_view.h"
#include "ava_reduce.h"


namespace stream::numerics {
    ConjugateGradient::ConjugateGradient(){
        d_r = AvaDeviceArray<fp_tt, int>::create({0});       // residue
        d_d = AvaDeviceArray<fp_tt, int>::create({0});       // search direction
        d_d1 = AvaDeviceArray<fp_tt, int>::create({0});      // matrix * search direction
        d_s = AvaDeviceArray<fp_tt, int>::create({0});       // preconditioned direction
        d_dots = AvaDeviceArray<fp_tt, int>::create({2});    // gpu alpha/beta values
        h_dots = AvaHostArray<fp_tt, int>::create({2});      // cpu alpha/beta values
        d_tmp_dot = AvaDeviceArray<fp_tt, int>::create({0}); // temp memory for dot product
        d_temp = AvaDeviceArray<fp_tt, int>::create({0});    // temp memory 
        temp_size = 0;
    }

    ConjugateGradient::~ConjugateGradient(){ }

    void ConjugateGradient::init(uint32_t n){
        d_r->resize({(int) n});
        d_d->resize({(int) n});
        d_d1->resize({(int) n});
        d_s->resize({(int) n});
        d_tmp_dot->resize({(int) n}); // Temp memory for dot product
        d_temp->resize({(int) n});    // Temp memory for CUB calls
        temp_size = n*sizeof(fp_tt);
    }

    void ConjugateGradient::solve(LinearSystem const * const system, AvaDeviceArray<fp_tt, int>::Ptr sol){
        AvaView<uint32_t, -1> d_col_v = system->d_csr.d_col->to_view<-1>();
        AvaView<uint32_t, -1> d_row_v = system->d_csr.d_row->to_view<-1>();
        AvaView<fp_tt, -1> d_A_v = system->d_csr.d_val->to_view<-1>();
        AvaView<fp_tt, -1> d_b_v = system->d_b->to_view<-1>();
        AvaView<fp_tt, -1> d_r_v = d_r->to_view<-1>();
        AvaView<fp_tt, -1> d_d_v = d_d->to_view<-1>();
        AvaView<fp_tt, -1> d_d1_v = d_d1->to_view<-1>();
        AvaView<fp_tt, -1> d_s_v = d_s->to_view<-1>();
        AvaView<fp_tt, -1> d_tmp_dot_v = d_tmp_dot->to_view<-1>(); // temp memory for dot product
        AvaView<fp_tt, -1> d_u_v = sol->to_view<-1>();

        uint32_t n = sol->size;

        // Init r and d
        ava_for<256>(nullptr, 0, n, [=] __device__ (uint32_t const tid){
            uint32_t const start = d_row_v(tid);
            uint32_t const end = d_row_v(tid+1);

            fp_tt dot = 0.0f;
            for (uint32_t i = start; i < end; i++){
                dot += d_A_v(i)*d_u_v(d_col_v(i));
            }

            d_r_v(tid) = d_b_v(tid) - dot;
            d_d_v(tid) = d_r_v(tid)/d_A_v(start);
            d_tmp_dot_v(tid) = d_r_v(tid)*d_d_v(tid);
        });

        ava::reduce::sum(d_temp->data, temp_size, d_tmp_dot->data, d_dots->data, n);
        deep_copy(h_dots->data(), d_dots->data, d_dots->size);

        uint32_t cg_iter = 0;
        while (sqrtf(h_dots(0)) > 1e-6f && cg_iter++ < 10000){

            // Compute d1 = Ad
            ava_for<256>(nullptr, 0, n, [=] __device__ (uint32_t const tid){
                uint32_t const start = d_row_v(tid);
                uint32_t const end = d_row_v(tid+1);

                fp_tt dot = 0.0f;
                for (uint32_t i = start; i < end; i++){
                    dot += d_A_v(i)*d_d_v(d_col_v(i));
                }

                d_d1_v(tid) = dot;
                d_tmp_dot_v(tid) = d_d_v(tid)*d_d1_v(tid);
            });

            ava::reduce::sum(d_temp->data, temp_size, d_tmp_dot->data, d_dots->data+1, n);
            deep_copy(h_dots->data(), d_dots->data, d_dots->size);

            fp_tt const alpha = h_dots(0)/h_dots(1);
            ava_for<256>(nullptr, 0, n, [=] __device__ (uint32_t const tid){ 
                d_u_v(tid) += alpha*d_d_v(tid);
                d_r_v(tid) -= alpha*d_d1_v(tid);
                d_s_v(tid) = d_r_v(tid)/d_A_v(d_row_v(tid));
                d_tmp_dot_v(tid) = d_r_v(tid)*d_s_v(tid);
            });

            fp_tt const old_rTr = h_dots(0);
            ava::reduce::sum(d_temp->data, temp_size, d_tmp_dot->data, d_dots->data, n);
            deep_copy(h_dots->data(), d_dots->data, d_dots->size);

            fp_tt const beta = h_dots(0)/old_rTr;
            ava_for<256>(nullptr, 0, n, [=] __device__ (uint32_t const tid){ 
                d_d_v(tid) = d_s_v(tid) + beta*d_d_v(tid);
            });
        }

        printf("Residue after %u iterations : %f\n", cg_iter, sqrtf(h_dots(0)));
    }
}
