
import torch as th
import numpy as np
from einops import rearrange


from pathlib import Path
import torchvision.io as tvio


import st_spix
from st_spix.flow_utils import run_raft,run_spynet

def read_video(root,ext="jpg"):
    name = "%05d." + ext
    vid = []
    # # minH,minW = 704,1008
    # minH,minW = 704,704
    start_index = -1
    for idx in range(101):
        fn = root / (name % idx)
        if not fn.exists(): continue
        if start_index < 0:
            start_index = idx
        img = tvio.read_image(fn)/255.
        # print(img.shape)
        # exit()
        F,H,W = img.shape
        # minH = H if H < minH else minH
        # minW = W if W < minW else minW
        vid.append(img)
    # for idx in range(len(vid)):
    #     vid[idx] = vid[idx][:,:minH,:minW]
    vid = th.stack(vid).cuda()
    # vid = resize(vid,(352,352)).to("cuda")
    return vid,start_index

def write_flo(flow, filename):
    """
    Writes a .flo file (optical flow) from a numpy array.

    Args:
        flow (numpy.ndarray): The optical flow array of shape (height, width, 2).
        filename (str): The path to save the .flo file.
    """

    flow = rearrange(flow,"two h w -> h w two").detach().cpu().numpy()
    with open(filename, 'wb') as f:
        # Write the header
        f.write(b'PIEH')  # Magic number
        f.write(np.array(flow.shape[1], dtype=np.int32).tobytes())  # Width
        f.write(np.array(flow.shape[0], dtype=np.int32).tobytes())  # Height
        # Write the flow data
        f.write(flow.astype(np.float32).tobytes())


def read_flo(filename):
    """
    Reads a .flo optical flow file and returns it as a numpy array.

    Args:
        filename (str): Path to the .flo file.

    Returns:
        numpy.ndarray: The optical flow array of shape (height, width, 2).
    """
    with open(filename, 'rb') as f:
        # Read the magic number and check its validity
        magic = f.read(4)
        if magic != b'PIEH':
            raise ValueError(f"Invalid .flo file: {filename}")

        # Read the width and height
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # Read the optical flow data
        flow_data = np.frombuffer(f.read(), dtype=np.float32)

        # Reshape the data to (height, width, 2)
        flow = flow_data.reshape((height, width, 2))

    return flow


def get_segtrackerv2_videos():
    root = Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/GroundTruth/")
    vid_names = list([v.name for v in root.iterdir()])
    # vid_names = ["frog_2","girl"]
    return vid_names

def write_flows(root,flows,start_index=0):
    import torchvision.io as tvio
    name = "%05d.flo"
    vid = []
    for idx in range(start_index,len(flows)+start_index):
        fn = root / (name % idx)
        write_flo(flows[idx-start_index],fn)

def viz_flows(flow_dir,viz_dir):
    if not viz_dir.exists(): viz_dir.mkdir(parents=True)
    import st_spix

    nfiles = len(list(flow_dir.iterdir()))
    # for flow_fn in flow_dir.iterdir():
    # for stem in ["00006"]:
    if "BIST" in str(viz_dir): start = 1
    else: start = 0
    for index in range(start,nfiles+start):
        # stem = flow_fn.stem
        stem = "%05d" % index
        flow_fn = flow_dir / ("%s.flo"%stem)
        flow = read_flo(flow_fn)
        viz_fn = viz_dir / ("%s.png"%flow_fn.stem)
        flow = rearrange(flow,'h w two -> 1 two h w')
        # print(flow_fn.stem)
        # if "00006" in flow_fn.stem:
        #     print(flow[0,:,48,34])
        #     print(flow[0,:,34,48])
        #     print(flow[0,:,49,76])
        #     exit()
        # print(flow_fn.stem,np.mean(np.abs(flow)))
        # flow = flow / (np.mean(np.abs(flow))+1e-10)
        # print(flow.shape)
        st_spix.flow_utils.viz_flow_quiver(viz_fn,flow,step=4)
        # save_flow(flow,viz_dir,flow_fn.name)

def main():

    # img_root=Path("/home/gauenk/Documents/packages/st_spix_refactor/tiny_video/images/")
    # flow_root = Path("/home/gauenk/Documents/packages/st_spix_refactor/tiny_video/flow/")
    flow_str = "RAFT_flows"
    # flow_str = "SPYNET_flows"
    vnames = get_segtrackerv2_videos()
    vnames = ["."]
    for vname in vnames:

        # -- config --
        img_root=Path("/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/PNGImages/%s/"%vname)
        img_root=Path("/home/gauenk/Documents/packages/st_spix_refactor/data/rep/")
        # flow_root = img_root / "RAFT_flows"
        flow_root = img_root / flow_str
        if not flow_root.exists():
            flow_root.mkdir()

        # -- viz --
        # viz_root = Path("./output/viz_flows/frog_2_raft")
        # flow_root = img_root / "BIST_flows"
        # viz_root = Path("./output/viz_flows/frog_2_bist")
        # viz_flows(flow_root,viz_root)
        # return

        # -- read video --
        # vid,start_index = read_video(img_root,"jpg")
        vid,start_index = read_video(img_root,"png")
        vid = vid[:,:3].contiguous()

        # -- run raft --
        if "raft" in flow_str.lower():
            fmax = 20
            fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
        elif "spynet" in flow_str.lower():
            fmax = 30
            fflow,bflow = run_spynet(vid)
        else:
            raise ValueError(".")

        # -- [dev] checking on spynet --
        # fflow,bflow = run_raft(th.clip(255.*vid,0.,255.).type(th.uint8))
        # fflow,bflow = run_raft(th.clip(255.*vid,0.,255.))
        # # _fflow,_bflow = run_spynet(2*(vid-0.5))
        # _fflow,_bflow = run_spynet(vid)

        # -- difference --
        # b_delta = th.abs(bflow[1:2] - _bflow[1:2]).ravel()
        # b_delta = th.quantile(b_delta,0.8).item()
        # print(b_delta)
        # b_delta = th.mean(th.abs(fflow - _bflow)).item()
        # # b_delta = th.mean(th.abs(bflow - _bflow)).item()
        # print(b_delta)
        # exit()
        # # f_delta = th.abs(fflow[1:2] - _fflow[1:2]).ravel()
        # # f_delta = th.quantile(f_delta,0.8).item()
        # # print(f_delta)
        #
        # print(bflow[1,0,100:103,100:103])
        # print(bflow[1,1,100:103,100:103])
        # print(_bflow[1,0,100:103,100:103])
        # print(_bflow[1,1,100:103,100:103])
        # exit()

        # -- clip for sanity --
        fflow = th.clip(fflow,-fmax,fmax)
        bflow = th.clip(bflow,-fmax,fmax)
        # print(fflow[0,:,0,0])
        # print(fflow[0,:,0,1])
        # print(fflow[0,:,1,0])
        # print(fflow[0,:,1,1])
        # print(fflow[0,:,0,0])
        # print(fflow[1,:,0,0])
        # print("fflow.shape: ",fflow.shape)
        # print("bflow.shape: ",bflow.shape)

        # -- write --
        # write_flows(flow_root,fflow)
        write_flows(flow_root,-bflow,start_index)


if __name__ == "__main__":
    main()
