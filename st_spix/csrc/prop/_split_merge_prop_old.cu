
__global__ void calc_bn_merge_p(int* seg, int* sm_pairs,
                              spix_params* sp_params,
                              spix_helper* sp_helper,
                              spix_helper_sm* sm_helper,
                              const int npix, const int nbatch,
                              const int width, const int nspix_buffer, float b_0) {

    // todo -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    // TODO: check if there is no neigh
    //get the label of neigh
    int f = sm_pairs[2*k+1];
	//if (sp_params[f].valid == 0) return;
    //if (f<=0) return;

    // -- unpack --

    float count_f = __ldg(&sp_params[f].count);
    float count_k = __ldg(&sp_params[k].count);

    float squares_f_x = __ldg(&sp_helper[f].sq_sum_app.x);
    float squares_f_y = __ldg(&sp_helper[f].sq_sum_app.y);
    float squares_f_z = __ldg(&sp_helper[f].sq_sum_app.z);
   
    float squares_k_x = __ldg(&sp_helper[k].sq_sum_app.x);
    float squares_k_y = __ldg(&sp_helper[k].sq_sum_app.y);
    float squares_k_z = __ldg(&sp_helper[k].sq_sum_app.z);

    float mu_f_x = __ldg(&sp_helper[f].sum_app.x);
    float mu_f_y = __ldg(&sp_helper[f].sum_app.y);
    float mu_f_z = __ldg(&sp_helper[f].sum_app.z);
   
    float mu_k_x = __ldg(&sp_helper[k].sum_app.x);
    float mu_k_y = __ldg(&sp_helper[k].sum_app.y);
    float mu_k_z = __ldg(&sp_helper[k].sum_app.z);

    int count_fk = count_f + count_k;

    // -- split merge --
    sm_helper[k].count_f = count_fk;
    sm_helper[k].b_n_app.x = b_0 + 0.5 * (squares_k_x -( mu_k_x*mu_k_x/count_k));
    sm_helper[k].b_n_f_app.x = b_0 + 0.5 *( (squares_k_x+squares_f_x) -
                              ( (mu_f_x + mu_k_x ) * (mu_f_x + mu_k_x ) / (count_fk)));
    sm_helper[k].b_n_app.y = b_0 + 0.5 * ((squares_k_y) -( mu_k_y*mu_k_y/count_k));
    sm_helper[k].b_n_f_app.y = b_0 + 0.5 *( (squares_k_y+squares_f_y) -
                              ( (mu_f_y + mu_k_y ) * (mu_f_y + mu_k_y ) / (count_fk)));
    sm_helper[k].b_n_app.z = b_0 + 0.5 * ((squares_k_z) -( mu_k_z*mu_k_z/count_k));
    sm_helper[k].b_n_f_app.z = b_0 + 0.5 *( (squares_k_z+squares_f_z) -
                              ( (mu_f_z + mu_k_z ) * (mu_f_z + mu_k_z ) / (count_fk)));

    // -- no negatives --
    if(  sm_helper[k].b_n_app.x<0)   sm_helper[k].b_n_app.x = 0.1;
    if(  sm_helper[k].b_n_app.y<0)   sm_helper[k].b_n_app.y = 0.1;
    if(  sm_helper[k].b_n_app.z<0)   sm_helper[k].b_n_app.z = 0.1;
    if(  sm_helper[k].b_n_f_app.x<0)   sm_helper[k].b_n_f_app.x = 0.1;
    if(  sm_helper[k].b_n_f_app.y<0)   sm_helper[k].b_n_f_app.y = 0.1;
    if(  sm_helper[k].b_n_f_app.z<0)   sm_helper[k].b_n_f_app.z = 0.1;

}


__global__ void calc_bn_split_p(int* sm_pairs,
                                spix_params* sp_params,
                                spix_helper* sp_helper,
                                spix_helper_sm* sm_helper,
                                int oldnew_direction,
                                const int npix, const int nbatch,
                                const int width, const int nspix_buffer,
                                float b_0, float sigma2_app, int max_nspix){

// __global__ void calc_bn_split_p(int* sm_pairs,
//                                 spix_params* sp_params,
//                                 spix_helper* sp_helper,
//                                 spix_helper_sm* sm_helper,
//                                 int oldnew_direction,
//                                 const int npix, const int nbatch,
//                                 const int width, const int nspix_buffer,
//                                 float b_0, int sigma2_app, int max_nspix) {
  // this is b_n = b_0+...
  // todo; -- add nbatch
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nspix_buffer) return;
	if (sp_params[k].valid == 0) return;
    // TODO: check if there is no neigh
    //get the label of neigh

    // -- split --
    int s = k + max_nspix;
	if (s>=nspix_buffer) return;
    // float count_f = __ldg(&sp_params[k].count);
    // float count_k = __ldg(&sm_helper[k].count);
    // float count_s = __ldg(&sm_helper[s].count);
    int count_f = __ldg(&sp_params[k].count);
    int count_k = __ldg(&sm_helper[k].count);
    int count_s = __ldg(&sm_helper[s].count);

    if((count_f<1)||( count_k<1)||(count_s<1)) return;


    // int count_prior = sp_params[k].prior_kappa;
    // int count_tot_k = count_k + count_prior;
    // double2 prior_mu_shape;
    // float3 prior_mu_app;
    // int count_pr = sm_params[k].prior_count;
    // int count_pr_app = sm_params[k].prior_mu_app_count;

    // -- load prior info --
    // float mu_pr_x = mu_pr.x;
    // float mu_pr_y = mu_pr.y;
    // float mu_pr_z = mu_pr.z;

    // int _count_pr_app_k = sp_params[k].prior_mu_app_count;
    // int _count_pr_app_s = sp_params[s].prior_mu_app_count;
    // int count_tot_s = count_s + _count_pr_app_s;
    // int count_tot_k = count_k + _count_pr_app_k;
    // int count_tot_f = count_f + _count_pr_app_k;
    // float count_pr_app = (float)_count_pr_app;

    // float squares_s_x = __ldg(&sm_helper[s].sq_sum_app.x);
    // float squares_s_y = __ldg(&sm_helper[s].sq_sum_app.y);
    // float squares_s_z = __ldg(&sm_helper[s].sq_sum_app.z);
   
    // float squares_k_x = __ldg(&sm_helper[k].sq_sum_app.x);
    // float squares_k_y = __ldg(&sm_helper[k].sq_sum_app.y);
    // float squares_k_z = __ldg(&sm_helper[k].sq_sum_app.z);
   


    // float sum_s_x = __ldg(&sm_helper[s].sum_app.x);
    // float sum_s_y = __ldg(&sm_helper[s].sum_app.y);
    // float sum_s_z = __ldg(&sm_helper[s].sum_app.z);

    // float sum_k_x = __ldg(&sm_helper[k].sum_app.x);
    // float sum_k_y = __ldg(&sm_helper[k].sum_app.y);
    // float sum_k_z = __ldg(&sm_helper[k].sum_app.z);

    // float sum_f_x = sum_s_x + sum_k_x;
    // float sum_f_y = sum_s_y + sum_k_y;
    // float sum_f_z = sum_s_z + sum_k_z;

    /********************
  
          Appearance
   
    **********************/

    float3 mu_pr_k = sp_params[k].prior_mu_app;
    float3 mu_pr_f = mu_pr_k;
    sp_params[s].prior_mu_app.x = 0;
    sp_params[s].prior_mu_app.y = 0;
    sp_params[s].prior_mu_app.z = 0;
    float3 mu_pr_s = sp_params[s].prior_mu_app;

    sp_params[s].prior_mu_app_count = 1;
    int prior_mu_app_count_s = sp_params[s].prior_mu_app_count;
    int prior_mu_app_count_k = sp_params[k].prior_mu_app_count;
    int prior_mu_app_count_f = prior_mu_app_count_k;

    double3 sum_s = sm_helper[s].sum_app;
    double3 sum_k = sm_helper[k].sum_app;
    double3 sum_f;
    sum_f.x = sum_s.x + sum_k.x;
    sum_f.y = sum_s.y + sum_k.y;
    sum_f.z = sum_s.z + sum_k.z;

    double3 sq_sum_s = sm_helper[s].sum_app;
    double3 sq_sum_k = sm_helper[k].sum_app;
    double3 sq_sum_f;
    sq_sum_f.x = sq_sum_s.x + sq_sum_k.x;
    sq_sum_f.y = sq_sum_s.y + sq_sum_k.y;
    sq_sum_f.z = sq_sum_s.z + sq_sum_k.z;

    // DONT USE ME I AM NOT CALLED!!
    double lprob_k = marginal_likelihood_app_sm(sum_k,sq_sum_k,mu_pr_k,count_k,
                                                prior_mu_app_count_k,sigma2_app);
    double lprob_s = marginal_likelihood_app_sm(sum_s,sq_sum_s,mu_pr_s,count_s,
                                                prior_mu_app_count_k,sigma2_app);
    double lprob_f = marginal_likelihood_app_sm(sum_f,sq_sum_f,mu_pr_f,count_f,
                                               prior_mu_app_count_k,sigma2_app);

    // -- write --
    sm_helper[k].numerator_app = lprob_k;
    sm_helper[s].numerator_app = lprob_s;
    sm_helper[k].numerator_f_app = lprob_f;

    // "left"
    // sm_helper[k].b_n_app.x = b_0+0.5*(mu_pr_k.x*mu_pr_k.x*count_pr_app/2 + squares_k_x - ( sum_k_x*sum_k_x/ count_tot_k));
    // sm_helper[k].b_n_app.y = b_0+0.5*(mu_pr_k.y*mu_pr_k.y*count_pr_app/2 + squares_k_y - ( sum_k_y*sum_k_y/ count_tot_k));
    // sm_helper[k].b_n_app.z = b_0+0.5*(mu_pr_k.z*mu_pr_k.z*count_pr_app/2 + squares_k_z - ( sum_k_z*sum_k_z/ count_tot_k));

    // "right"
    // sm_helper[s].b_n_app.x = b_0 + 0.5*((squares_s_x) - ( sum_s_x*sum_s_x/ count_tot_s));
    // sm_helper[s].b_n_app.y = b_0 + 0.5*((squares_s_y) - ( sum_s_y*sum_s_y/ count_tot_s));
    // sm_helper[s].b_n_app.z = b_0 + 0.5*((squares_s_z) - ( sum_s_z*sum_s_z/ count_tot_s));

    // "full"
    // sm_helper[k].b_n_f_app.x = b_0 + 0.5 * (mu_pr_k.x*mu_pr_k.x*count_pr_app + (squares_k_x+squares_s_x) - ( sum_f_x*sum_f_x/ count_tot_f));
    // sm_helper[k].b_n_f_app.y = b_0 + 0.5 * (mu_pr_k.y*mu_pr_k.y*count_pr_app + (squares_k_y+squares_s_y) - ( sum_f_y*sum_f_y/ count_tot_f));
    // sm_helper[k].b_n_f_app.z = b_0 + 0.5 * (mu_pr_k.z*mu_pr_k.z*count_pr_app + (squares_k_z+squares_s_z) - ( sum_f_z*sum_f_z/ count_tot_f));

    // __device__ double marginal_likelihood_app_sm(double2 sum_obs,double2 sq_sum_obs,
    //                                            double2 prior_mu,int _num_obs,
    //                                            int _num_prior, double sigma2) {

    // marginal_likelihood_app_sm(double2 sum_obs,double2 sq_sum_obs,
    //                                                double2 prior_mu,int _num_obs,
    //                                                int _num_prior, double sigma2)


    /********************
  
             Shape
   
    **********************/

    // -- read statistics --
    int prior_count = sp_params[k].prior_count;
    int prior_mu_shape_count_k = sp_params[k].prior_mu_shape_count;
    int prior_mu_shape_count_f = sp_params[k].prior_mu_shape_count;
    int prior_sigma_shape_count = sp_params[k].prior_sigma_shape_count;
    double3 prior_sigma_shape_k = sp_params[k].prior_sigma_shape;
    double3 prior_sigma_shape_f = sp_params[k].prior_sigma_shape;
    double2 prior_mu_shape_k = sp_params[k].prior_mu_shape;
    double2 prior_mu_shape_f = sp_params[k].prior_mu_shape;

    // -- new --
    sp_params[s].prior_mu_app.x = 0;
    sp_params[s].prior_mu_app.y = 0;
    sp_params[s].prior_mu_app.z = 0;
    float3 prior_mu_app_s = sp_params[s].prior_mu_app;
    // sp_params[s].prior_mu_app_count = 1;
    // int prior_mu_app_count_s = sp_params[s].prior_mu_app_count;
    sp_params[s].prior_mu_shape.x = 0;
    sp_params[s].prior_mu_shape.y = 0;
    double2 prior_mu_shape_s = sp_params[s].prior_mu_shape;
    sp_params[s].prior_mu_shape_count = 1;
    int prior_mu_shape_count_s = sp_params[s].prior_mu_app_count;
    sp_params[s].prior_sigma_shape.x = prior_count*prior_count;
    sp_params[s].prior_sigma_shape.y = 0;
    sp_params[s].prior_sigma_shape.z = prior_count*prior_count;
    double3 prior_sigma_shape_s = sp_params[s].prior_sigma_shape;
    sp_params[s].prior_sigma_shape_count = prior_count;
    int prior_sigma_shape_count_s = sp_params[s].prior_sigma_shape_count;


    int2 sum_shape_k = sp_helper[k].sum_shape;
    int2 sum_shape_s = sp_helper[s].sum_shape;
    int2 sum_shape_f;
    sum_shape_f.x = sum_shape_k.x+sum_shape_s.x;
    sum_shape_f.y = sum_shape_k.y+sum_shape_s.y;
    longlong3 sq_sum_shape_k = sm_helper[k].sq_sum_shape;
    longlong3 sq_sum_shape_s = sm_helper[s].sq_sum_shape;
    longlong3 sq_sum_shape_f;
    sq_sum_shape_f.x = sq_sum_shape_k.x + sq_sum_shape_s.x;
    sq_sum_shape_f.y = sq_sum_shape_k.y + sq_sum_shape_s.y;
    sq_sum_shape_f.z = sq_sum_shape_k.z + sq_sum_shape_s.z;

    // -- sample stats --
    double2 mu_shape_k = calc_shape_sample_mean_sm(sum_shape_k,count_k);
    double2 mu_shape_s = calc_shape_sample_mean_sm(sum_shape_s,count_s);
    double2 mu_shape_f = calc_shape_sample_mean_sm(sum_shape_f,count_f);

    double3 sigma_k = calc_shape_sigma_mode_sm(sq_sum_shape_k,mu_shape_k,
                                               prior_sigma_shape_k,
                                               prior_mu_shape_k,count_k,
                                               prior_mu_shape_count_k);
    double3 sigma_s = calc_shape_sigma_mode_sm(sq_sum_shape_s,mu_shape_s,
                                               prior_sigma_shape_s,
                                               prior_mu_shape_s,count_s,
                                               prior_mu_shape_count_s);
    double3 sigma_f = calc_shape_sigma_mode_sm(sq_sum_shape_f,mu_shape_f,
                                               prior_sigma_shape_f,
                                               prior_mu_shape_f,count_f,
                                               prior_mu_shape_count_f);
    // mu_shape_k = calc_shape_mean_mode_sm(mu_shape_k,prior_mu_shape_k,
    //                                      count_k,prior_mu_shape_count);
    // mu_shape_s = calc_shape_mean_mode_sm(mu_shape_s,prior_mu_shape_s,
    //                                      count_s,prior_mu_shape_count);
    // mu_shape_f = calc_shape_mean_mode_sm(mu_shape_f,prior_mu_shape_f,
    //                                      count_f,prior_mu_shape_count);


    // compute determinants
    sm_helper[k].b_n_shape_det = determinant2x2_sm(sigma_k); // left
    sm_helper[s].b_n_shape_det = determinant2x2_sm(sigma_s); // right
    sm_helper[s].b_n_f_shape_det = determinant2x2_sm(sigma_f); //full

}


