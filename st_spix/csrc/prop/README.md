This is an outline of the files.


Notes:

-> we need to initialize the superpixel parameters, including "prior_count". see "sp_helper.cu" for more details.
-> 

Major Code Files:

- "fill_missing.cu": Fills overlaps/holes after rigidly shifting superpixel segmentations across time. This only uses the spatial nearest neighbor and does not use any parameters. It is a fast initialization of the new state.

- "split_disconnected.cu": This code splits disconnected regions. Regions can split due to video dynamics, but the newly disconnected regions require their own parameters.

- "refine_missing.cu": Updates the boarder along only the missing pixels to improve the initial state before BASS (with splits/merges) are run.

- "bass_iters.cu": Runs the propogated BASS. Expects an intialized spix and using the previous means as the prior for the current means

Helping Code Files:

- "update_prop_params.cu": Update the parameters using the previous frames parameters. The parameters are "propgated" (hence 'prop') from the previous frame.

- "update_missing_seg.cu": Update the segmentation only if the border pixel was a missing pixel.

- "update_prop_seg.cu": Updates the segmentation as usual. Basically, a copy from BASS.

- "seg_utils.cu": Shared functions for update_[missing/prop]_seg.cu files.

Other Deps:
-> refine_missing:
   - "../bass/core/update_seg.cu"
   - 

Outline of Algorithm:

- shift superpixels (Python)
- fill_missing
- split_disconnected
- refine_missing (update_prop_params,update_missing_seg)
- bass_iters (update_prop_params,update_prop_seg)

Work so Far:

- [done] fill_missing
- [done] split_disconnected
- [almost draft] refine_missing
- [wip] bass_iters

- [draft] update_missing_seg
- [draft] update_prop_seg
- [draft] update_prop_params
