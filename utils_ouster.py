#from calendar import c
#from mimetypes import init
from pydoc import doc
from anyio import current_time
from ouster import client, pcap
from contextlib import closing, ExitStack
from datetime import datetime as dt
import cv2
import matplotlib.pyplot as plt
from more_itertools import time_limited,nth
import numpy as np
from timeit import default_timer as timer
from torch import dtype
from tqdm import tqdm
import time
from datetime import datetime as dt
from copy import copy
import open3d as o3d
import math
import os
LIMITS_IRR = {"ir":6000,"reflectivity": 255, "range":25000,"signal":255} #255 for IRR #128 for SRR¨
LIMITS_SRR = {"ir":6000,"reflectivity": 128, "range":25000,"signal":128}
def sensor_config(hostname = 'os-122107000535.local',lidar_port = 7502, imu_port = 7503,phase_lock =None): 
    """
    Set sensor configuration.
    @param hostname: sensor hostname
    @param lidar_port: lidar port
    @param imu_port: imu port

    @return: Sensor Config Object.
    """
   
    print(f"Configuring sensor: {hostname}")
    # establish sensor connection
    config = client.SensorConfig()
    # set the values that you need: see sensor documentation for param meanings
    config.operating_mode = client.OperatingMode.OPERATING_NORMAL
    config.lidar_mode = client.LidarMode.MODE_1024x20
    config.udp_port_lidar = lidar_port
    config.udp_port_imu = imu_port
    if phase_lock is not None:
        config.phase_lock_enable = True
        config.phase_lock_offset = phase_lock

    # set the config on sensor, using appropriate flags
    client.set_config(hostname, config, persist=True, udp_dest_auto=True)
    return [config, hostname]

def record_lidar_seq(
                    config=None,
                    fileformat="npy",
                    lidar_folder_path="../lidar_scans",
                    scan_name = str(f"Ouster_{dt.now().strftime('%Y%m%d_%H%M%S')}"),
                    set_frames = False,
                    seq_length = 1.0,
                    hostname = "192.168.200.78",#'os-122107000535.local',
                    lidar_port = 7502,
                    imu_port = 7503
                    ):
    """
    Record lidar data for n_seconds and save to local generated path..
    @param config: SensorConfig object
    @param fileformat: string (nly, npy, pcap)
    @param lidar_folder_path: string (path to lidar folder)
    @param scan_name: string (name of scan)
    @param set_frames: bool if True will only record a fixed number of frames. If False will record seq_lenght seconds
    @param seq_length: float (number of frames to record) if set_frames is False else number of frames to record.
    @param hostname: string
    @param lidar_port: int (default 7502)
    @param imu_port: int (default 7503)


    """
    if config is None:
        try:
            [config,_] = sensor_config(hostname=hostname,lidar_port=lidar_port,imu_port=imu_port)
        except:
            raise Exception("Could not connect to sensor.")
    if not os.path.exists(lidar_folder_path):
        os.makedirs(lidar_folder_path)
    scan_path = f"{lidar_folder_path}/{scan_name}/"
    if not os.path.exists(scan_path):
        os.mkdir(f"{lidar_folder_path}/{scan_name}")
    # connect to sensor and record lidar/imu packets
    frames = 0
    full_seq = []
    if fileformat == "npy":
        
        with closing(client.Scans.stream(hostname, lidar_port,
                                    complete=False)) as stream:
            print(f"Record lidar data to npy files {seq_length}:")
            with tqdm(total=seq_length) as pbar:
                start = time.time()
                for seq,scan in enumerate(stream,0):
                    xyz = get_xyz(stream,scan)
                    #print(f"xyz shape: {xyz.shape}")
                    signal = get_signal_reflection(stream,scan)
                    #print(f"xyz shape: {xyz.shape}")
                    xyzr = convert_to_xyzr(xyz,signal)
                    #print(f"xyz shape: {xyz.shape}")
                    comp_xyzr = compress_mid_dim(xyzr)
                    #print(f"comp_xyzr shape: {comp_xyzr.shape}")
                    full_seq.append(comp_xyzr)
                    save_to_nly(comp_xyzr,scan_path,seq)
                    #print(f"Time: {time.time()-start} frame {seq}")
                    if not set_frames:
                        p = min(((time.time()-start)/seq_length),1)
                        pbar.update(p)
                        if time.time()-start>seq_length:
                            break
                    if seq>seq_length and set_frames:
                        break
            full_seq = np.asarray(full_seq)
            print(f"Full Seq shape: {full_seq.shape}")
            save_to_nly(full_seq,f"{scan_path}/FS",0)
    
    elif fileformat == "pcap":
        with closing(client.Sensor(hostname, lidar_port, imu_port,
                                   buf_size=640)) as source:

            # make a descriptive filename for metadata/pcap files

            time_part = dt.now().strftime("%Y%m%d_%H%M%S")
            meta = source.metadata
            fname_base = f"{meta.prod_line}_{meta.sn}_{meta.mode}_{time_part}"

            print(f"Saving sensor metadata to: {fname_base}.json")
            source.write_metadata(f"{fname_base}.json")

            print(f"Writing to: {fname_base}.pcap (Ctrl-C to stop early)")
            source_it = time_limited(seq_length, source)
            n_packets = pcap.record(source_it, f"{fname_base}.pcap")
            print(f"Captured {n_packets} packets")
    return scan_path,frames

def save_to_nly(xyzr,seq_path,seq):
    """
    Save lidar data to nly file.
    @param xyzr: numpy array
    @param scan_folder: string (name of scan folder)
    @param seq: int (sequence number)
    """
    filename = f"{seq_path}{seq}"
    #print(f"Saving to: {filename}")
    np.save(filename,xyzr)

def unpack_multi_dim(scan_path):
    """ NOT WORKING YET (!!!)
    Unpack lidar data from multi-dim array.
    @param scan_path: string (path to lidar scans) 
    """
    scan_path = None
    frame_id = 0
    lst = os.listdir(scan_path)
    print(lst)
    lst.sort()
    print(lst)
    for i,dirs in enumerate(lst):
        if dirs.endswith(".npy"):
            N_xyzr = np.load(f"{scan_path}/{dirs}")
            load_path = f"{scan_path}/{dirs}"
            if len(N_xyzr.shape) == 2:
                save_to_nly(N_xyzr,scan_path,frame_id)
                frame_id += 1
            elif len(N_xyzr.shape) == 3:
                for xyzr in N_xyzr:
                    save_to_nly(xyzr,load_path,frame_id)
                    frame_id += 1
            else:
                print(f"Ignoring file: {dirs}. Wrong dimensions.")
        elif os.path.isdir(dirs):
            unpack_multi_dim(f"{scan_path}/{dirs}")
import re
def sorted_alphanumeric(data):
    """
    Sort alphanumeric strings.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    
def trim_xyzr(xyzr,trim_thresh):
    """
    Trim lidar data to remove points that are too close to the sensor.
    @param xyzr: numpy array
    @param trim_thresh: float (threshold distance in meters)
    """
    #indices = np.where(xyzr[:,0]>trim_thresh[0] and xyzr[:,1]>trim_thresh[1] and xyzr[:,2]>trim_thresh[2])
    return xyzr[(abs(xyzr[:,0])<trim_thresh[0]) * (abs(xyzr[:,1])<trim_thresh[1]) * (abs(xyzr[:,2])<trim_thresh[2]),:]
def play_back_recording(seq_dir_path,trim=None):
    """
    Play back lidar data from nly files.
    @param seq_dir_path: string (path to lidar scan folder) containing .npy
    @param trim: [x,y,z] max
    """
    if not os.path.exists(seq_dir_path):
        raise FileNotFoundError(f"{seq_dir_path} does not exist")
    print(f"Playing back lidar data from: {seq_dir_path}")
    dirlist = sorted_alphanumeric(os.listdir(seq_dir_path))
    for i,files in enumerate(dirlist):
        if files.endswith(".npy") and not files.startswith("FS"):
            xyzr = np.load(f"{seq_dir_path}/{files}")
            print(f"Frame: {files[:-4]}")
            if len(xyzr.shape)==3:
                xyzr = compress_mid_dim(xyzr)
            if trim is not None:
                xyzr = trim_xyzr(xyzr,trim)
            xyz = xyzr[:,:3]
            signal = xyzr[:,3]
            #print(f"xyzr shape: {xyzr.shape}")
            if i == 0:
                vis, geo = initialize_o3d_plot(xyzr)
            else:
                update_open3d_live(geo, xyzr,vis)
                time.sleep(0.15)
            #time.sleep(0.1)
    
def play_back_recording_multifile(seq_dir_path):
    """
    Play back lidar data from nly files.
    @param seq_dir_path: string (path to lidar scan folder) containing .npy
    """
    if not os.path.exists(seq_dir_path):
        raise FileNotFoundError(f"{seq_dir_path} does not exist")
    print(f"Playing back lidar data from: {seq_dir_path}")
    for i,files in enumerate(os.listdir(seq_dir_path)):
        if files.endswith(".npy") and not files.startswith("FS"):
            print(f"Plotting lidar data from file: {files}")
            full_seq = np.load(f"{seq_dir_path}/{files}")
            for j,xyzr in enumerate(full_seq):
                xyzr = np.squeeze(xyzr)
                xyzr = compress_mid_dim(xyzr) if len(xyzr.shape)==3 else xyzr
                #print(f"xyzr shape: {xyzr.shape}")
                if j == 0:
                    vis, geo = initialize_o3d_plot(xyzr)
                else:
                    update_open3d_live(geo, xyzr,vis)
                    time.sleep(1)
            #time.sleep(0.1)
    
def process_lidar_scan(scan_path=None):
    """
    Process lidar scan data.
    @param scan_path: string (path to lidar scan folder) containing .npy
    """
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"{scan_path} does not exist")
    print(f"Processing lidar data from: {scan_path}")
    for i,files in enumerate(os.listdir(scan_path)):
        if files.endswith("FS.npy"):
            print(f"Processing lidar data from file: {files}")
            scn = np.load(f"{scan_path}/{files}")

def offset_pc(xyz:np.ndarray, positions:list):
    """
    Offset point cloud.
    @param xyz: numpy array (Nx3)
    @param positions: list of [x,y,z]
    """
    xyz = np.add(xyz,positions)
    return xyz
def combine_clouds(xyzr:list):
    """
    Combine point cloud and lidar scan.
    @param xyzr: list[numpy array (Nx4)]
    """
    xyzr = np.concatenate(xyzr,axis=0)
    return xyzr
def stream_from_multiple(args):
    """
    Stream lidar data from multiple ouster sensors.
    """
    lidar_ports = args.lidar_port
    imu_ports = args.imu_port
    hostnames = args.hostname if len(args.hostname)>0 else [None for ip in args.host_ip]
    host_ips = args.host_ip
    phase_locking = args.phase_locking
    stream_time = args.stream_time
    offsets = args.relative_position
    print(offsets)

    offsets = [[float(x) for x in lst]+[0.0] for lst in offsets]
    
    # print(f"Offsets: {offsets},shape: {offsets.shape}")
    # offsets = np.append(offsets,np.zeros((1,1)),axis=0)
    configs = []
    for i,(hostname,host_ip,lidar_port,imu_port) in enumerate(zip(hostnames,host_ips,lidar_ports,imu_ports)):
        print(f"Streaming from {hostname}:Lidar port {lidar_port}, IMU port {imu_port}")
        if hostname is not None:
            configs.append(sensor_config(hostname=hostname,lidar_port=int(lidar_port),imu_port=int(imu_port)))
        elif host_ip is not None:
            configs.append(sensor_config(hostname=host_ip,lidar_port=int(lidar_port),imu_port=int(imu_port)))
            hostname[i] = host_ip
    with ExitStack() as stack:
        streams = [stack.enter_context(closing(client.Scans.stream(hostname, int(lidar_port), complete=False))) for hostname, lidar_port in zip(hostnames,lidar_ports)]
        start_time = time.monotonic()
        clouds = []
        while time.monotonic()-start_time<stream_time:
            # scan_64 = next(iter(streams[0]))
            # scan_128 = next(iter(streams[1]))
            # xyz_64 = get_xyz(streams[0],scan_64)
            # xyz_128 = get_xyz(streams[1],scan_128)
            # signal_64 = get_signal_reflection(streams[0],scan_64)
            # signal_128 = get_signal_reflection(streams[1],scan_128)
            # xyzr_64 = convert_to_xyzr(xyz_64,signal_64)
            # xyzr_128 = convert_to_xyzr(xyz_128,signal_128)
            # comp_xyzr_64 = compress_mid_dim(xyzr_64)
            # comp_xyzr_128 = compress_mid_dim(xyzr_128)
            # comp_xyzr_64 = trim_xyzr(comp_xyzr_64,[4,4,4])
            # comp_xyzr_128 = trim_xyzr(comp_xyzr_128,[4,4,4])
            # comp_xyzr_64 = offset_pc(comp_xyzr_64,offsets[0])
            # comp_xyzr_128 = offset_pc(comp_xyzr_128,offsets[1])
            # clouds = [comp_xyzr_64,comp_xyzr_128]
            # for i,stream in enumerate(streams):
            #     # print("Streaming from:",hostnames[i])
            #     scan_time = time.monotonic()
            #     scan = next(iter(stream))
            #     print(f"Scan time {time.monotonic()-scan_time:.3e}")
            #     xyz = get_xyz(stream,scan)
            #     signal = get_signal_reflection(stream,scan)
            #     xyzr = convert_to_xyzr(xyz,signal)
            #     comp_xyzr = compress_mid_dim(xyzr)
            #     comp_xyzr = trim_xyzr(comp_xyzr,[4,4,4])
            #     comp_xyzr = offset_pc(comp_xyzr,offsets[i])
            #     clouds.append(comp_xyzr)
            # print(f"Streaming time: {time.monotonic()-start}")
            for i,(scan_64,scan_128) in enumerate(zip(streams[0],streams[1])):
                if i%2==0:
                    start = time.monotonic()
                else:
                    print(f"Streaming time: {time.monotonic()-start:.3e}")
                xyz_64 = get_xyz(streams[0],scan_64)
                xyz_128 = get_xyz(streams[1],scan_128)
                signal_64 = get_signal_reflection(streams[0],scan_64)
                signal_128 = get_signal_reflection(streams[1],scan_128)
                xyzr_64 = convert_to_xyzr(xyz_64,signal_64)
                xyzr_128 = convert_to_xyzr(xyz_128,signal_128)
                comp_xyzr_64 = compress_mid_dim(xyzr_64)
                comp_xyzr_128 = compress_mid_dim(xyzr_128)
                comp_xyzr_64 = trim_xyzr(comp_xyzr_64,[4,4,4])
                comp_xyzr_128 = trim_xyzr(comp_xyzr_128,[4,4,4])
                comp_xyzr_64 = offset_pc(comp_xyzr_64,offsets[0])
                comp_xyzr_128 = offset_pc(comp_xyzr_128,offsets[1])
                clouds = [comp_xyzr_64,comp_xyzr_128]
                combined_cloud = combine_clouds(clouds)
                if i==0:
                    vis, geo = initialize_o3d_plot(combined_cloud)
                else:
                    update_open3d_live(geo,combined_cloud,vis)
                print(f"Combined cloud shape: {combined_cloud.shape}")
def stream_live(args):
    """
    Stream Live from sensor belonging to hostname, given a specified config.
    @param config: SensorConfig object
    @param hostname: string
    @param lidar_port: int (default 7502)
    @param imu_port: int (default 7503)
    """
    lidar_port = args.lidar_port
    imu_port = args.imu_port
    hostname = args.hostname
    host_ip = args.host_ip
    frames_to_record = args.frames_to_record
    #plt.ion()
    if host_ip is not None:
        config = sensor_config(hostname=host_ip,lidar_port=lidar_port,imu_port=imu_port)
        hostname = host_ip
    elif hostname is not None:
        config = sensor_config(hostname=hostname,lidar_port=lidar_port,imu_port=imu_port)
    # create a stream object
    print("Start Lidar Stream:")
    with closing(client.Scans.stream(hostname, lidar_port,
                                    complete=False)) as stream:
        first_scan = next(iter(stream))
        xyz = get_xyz(stream,first_scan)
        signal = get_signal_reflection(stream,first_scan)
        xyzr = convert_to_xyzr(xyz,signal)
        comp_xyzr = compress_mid_dim(xyzr)
        comp_xyzr = trim_xyzr(comp_xyzr,[4,4,4])
        vis, geo = initialize_o3d_plot(comp_xyzr)
        plt.ion()
        # start the stream
        for i,scan in enumerate(stream):
            # uncomment if you'd like to see frame id printed
            # print("frame id: {} ".format(scan.frame_id))
            signal = get_signal_reflection(stream,scan)
            xyz = get_xyz(stream,scan)
            xyzr = convert_to_xyzr(xyz,signal)
            comp_xyzr = compress_mid_dim(xyzr)
            comp_xyzr = trim_xyzr(comp_xyzr,[LIMITS_IRR["range"]/1000,LIMITS_IRR["range"]/1000,LIMITS_IRR["range"]/1000])
            update_open3d_live(geo,comp_xyzr,vis)
            img = signal_ref_range(stream,scan,LIMITS_IRR)
            print(f"Average: \nsignal {np.mean(img[:,:,0])*LIMITS_IRR['signal']} \nReflectivity {np.mean(img[:,:,1])*LIMITS_IRR['reflectivity']}\nRange {np.mean(img[:,:,2])*LIMITS_IRR['range']} ")
            # if i == 0:
            #     #fig = plt.figure(figsize=(10,10))
            #     cv2.imshow("Lidar",img)
            imgsz = [1280,640]
            print(scan.h)
            cv2.imshow("Lidar",cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),imgsz))
            cv2.imshow("signal",cv2.resize(cv2.cvtColor(img[:,:,0],cv2.COLOR_BGR2RGB),imgsz))
            cv2.imshow("Range",cv2.resize(cv2.cvtColor(img[:,:,2],cv2.COLOR_BGR2RGB),imgsz))
            cv2.imshow("Reflectivity",cv2.resize(cv2.cvtColor(img[:,:,1],cv2.COLOR_BGR2RGB),imgsz))
            cv2.waitKey(1)
            
            
            # else:
            #     #plot_total.set_data(img)
            #     # plot_ref.set_data(np.array([img[:,:,0],np.zeros_like(img[:,:,0]),np.zeros_like(img[:,:,0])]).transpose(1,2,0))
            #     # plot_range.set_data(np.array([np.zeros_like(img[:,:,1]),img[:,:,1],np.zeros_like(img[:,:,1])]).transpose(1,2,0))
            #     # plot_ir.set_data(np.array([np.zeros_like(img[:,:,2]),np.zeros_like(img[:,:,2]),img[:,:,2]]).transpose(1,2,0))
            #     cv2.imshow("Lidar",img)

            #     fig.canvas.draw()
            #     fig.canvas.flush_events()
            #     time.sleep(0.001)
                # fig_ref.canvas.draw()
                # fig_ref.canvas.flush_events()
                # fig_range.canvas.draw()
                # fig_range.canvas.flush_events()
                # fig_ir.canvas.draw()
                # fig_ir.canvas.flush_events()

                # filename = f"../lidarImages/image_{i}.jpg"
                #if limits is None or (limits["reflectivity"] == 0 and limits["range"] == 0 and limits["ir"] == 0):
                #    layer_normalized_img =  np.array([(img[:,:,0])/img[:,:,0].max(),(img[:,:,1])/img[:,:,1].max(),(img[:,:,2])/img[:,:,2].max()])
                # cv2.imwrite(filename,cv2.cvtColor(img*255,cv2.COLOR_RGB2BGR))
                #plot.set_data(a)
            
            # print(f"T: {end-start} s")
            if i>=100000:
                break
def convert_to_xyzr(xyz,signal):
    """
    Convert lidar data to xyzr.
    @param xyz: numpy array of xyz data
    @param signal: numpy array of signal data

    @return: xyzr numpy array
    """
    if xyz is None:
        raise ValueError("xyz is None before concatenation with signal.")
    if signal is None:
        raise ValueError("signal is None before concatenation with xyz.")
    if len(xyz.shape) < 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz is not (channels,beams,3)")
    if len(signal.shape) < 3 or signal.shape[-1] != 1:
        #"Signals should same as xyz except final dim: (n_scans, n_points, 1)")
        signal = np.expand_dims(signal,axis=-1)
    #print("xyz.shape: {}".format(xyz.shape))
    #print("signal.shape: {}".format(signal.shape))
    try:
        xyzr = np.concatenate((xyz,signal),axis=-1)
    except:
        print("Different shapes for xyz and signal. xyz: {}, signal: {}".format(xyz.shape,signal.shape))
        return None
    return xyzr

def compress_mid_dim(data):
    """
    Compress mid dim of xyz,xyzr or signal to create (bs,n_points, 1 3 or 4).
    """
    if len(data.shape) < 3:
        raise ValueError(f"xyz should be: (channels, beams, 3 or 4). got {data.shape}.")
    return data.reshape(-1,data.shape[-1])

def trim_data(data,range_limit,source,scan):
    range_data = client.destagger(source.metadata,
                            scan.field(client.ChanField.RANGE))
    range_data = compress_mid_dim(np.expand_dims(range_data,axis=-1))
    #print(range_data.shape)
    indices = np.where(range_data < range_limit)[0].tolist()
    #print(len(indices))
    #print(data.shape)
    data = data[indices,:]
    #print(data.shape)
    return data
def get_signal_reflection(source,scan):
    """
    Get signal reflection from scan.
    @param scan: Scan object
    @param source: Source of data

    @return: numpy array of signal reflection
    """
    if scan is None:
        raise ValueError("scan is None")
    if source is None:
        raise ValueError("source is None")
    try:
        return client.destagger(source.metadata,
                            scan.field(client.ChanField.REFLECTIVITY))
    except:
        print("Error getting signal reflection.")
        return None
def get_xyz(source,scan,trim=None):
    """
    Get xyz data from scan.
    @param source: Source of data
    @param scan: Scan object
    @return: numpy array of xyz data
    """
    
    if source is None:
        raise ValueError("source is None")
    xyzlut = client.XYZLut(source.metadata)
    xyz = client.destagger(source.metadata, xyzlut(scan))

    #xyz  = xyzlut(scan)
    if trim is not None:
        indices = np.nonzero((sum((abs(xyz[:,:,0])<trim[0],abs(xyz[:,:,1])<trim[1],abs(xyz[:,:,2])<trim[2]))))[0].tolist()
         
        xyz = xyz[indices,:]
    return xyz

#def plot_lidar_example(xyz):
#    """
#    Plot lidar example from xyz data.
#    @param xyz: numpy array of xyz data
#    """
#    [x, y, z] = [c.flatten() for c in np.dsplit(xyz, 3)]
#    ax = plt.axes(projection='3d')
#    r = 5
#    ax.set_xlim3d([-r, r])
#    ax.set_ylim3d([-r, r])
#    ax.set_zlim3d([-r/2, r/2])
#    plt.axis('off')
#    z_col = np.minimum(np.absolute(z), 5)
#    ax.scatter(x, y, z, c=z_col, s=0.2)
#    plt.show()
def seperate_outliers(cloud,ind):
    """
    Seperate outliers from cloud.
    @param cloud: o3d cloud object
    @param ind: numpy array of indices of outliers

    @return: cloud_outliers: o3d cloud object of outliers
    @return: cloud_inliers: o3d cloud object of inliers
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    return inlier_cloud, outlier_cloud

def get_single_example():
    """
    Get a single example from the lidar data.

    """
    metadata_path = "/Users/theodorjonsson/GithubProjects/LidarStuff/RecordedData/OS0/json/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.json"
    pcap_path = "/Users/theodorjonsson/GithubProjects/LidarStuff/RecordedData/OS0/pcap/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.pcap"
    #pcap_path = '/Users/theodorjonsson/GithubProjects/ExampleData/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.pcap'
    #metadata_path = '/Users/theodorjonsson/GithubProjects/ExampleData/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.json'
    with open(metadata_path,'r') as file:
        info = client.SensorInfo(file.read())
    source = pcap.Pcap(pcap_path,info)
    with closing(client.Scans(source)) as scans:
        scan = nth(scans, 50) # Five second scan (50Rot/10Hz)
    return scan,source
def initialize_o3d_plot(xyzr):
    """
    Initialize o3d plot.
    @param xyzr: numpy array of xyzr data to initailize plot with.

    @return: o3d.visualization.Visualizer object
    """
    xyzr = xyzr[0] if len(xyzr.shape) > 2 else xyzr
    xyz = xyzr[:,:3]
    print(xyz)
    signal = xyzr[:,3:]
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geo = o3d.geometry.PointCloud()
    #print("xyz.shape: {}".format(xyz.shape))
    geo.points = o3d.utility.Vector3dVector(xyz)
    vis.add_geometry(geo)
    return vis,geo

def update_open3d_live(geo,xyzr,vis):
    """
    Plot lidar data live using open3d.
    @param xyzr: numpy array of xyzr data
    """
    xyzr = xyzr[0] if len(xyzr.shape) > 2 else xyzr

    xyz = xyzr[:,:3]
    signal = xyzr[:,3:]
    #print("xyz.shape: {}".format(xyz.shape))
    geo.points = o3d.utility.Vector3dVector(xyz)
    vis.update_geometry(geo)
    vis.poll_events()
    vis.update_renderer()
    
    
def plot_open3d_pc(xyzr):
    """
    Plot lidar example from xyzr data.
    @param xyzr: numpy array of xyzr data. Each cloud is stacked in dim=0. Iterate over dim=0 to visualize all clouds.
    """
    xyz = xyzr[:,:,:3] 
    signal = xyzr[:,:,3:] # 3: for broadcasting
    print("Shape of xyz: {}".format(xyz.shape))
    print("Shape of signal: {}".format(signal.shape))
   
    vis,geo = initialize_o3d_plot(xyz[0])
    for i,cloud in enumerate(xyz[1:],1):
        geo.points = o3d.utility.Vector3dVector(cloud)
        geo.colors = o3d.utility.Vector3dVector(np.zeros((xyzr.shape[1],3))+signal[i,:,:]) # Not finished
        vis.update_geometry(geo)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1) # Time between clouds

def ir_ref_range(source,scan):
    
    near_ir = client.destagger(source.metadata,scan.field(client.ChanField.NEAR_IR))
    if LIMITS_IRR["ir"] not in [0,None]:
        near_ir = np.divide(near_ir,LIMITS_IRR["ir"], dtype=np.float32)
        near_ir = np.where(near_ir<1,near_ir,1)
    # print(f"max near_ir: {near_ir.max()} at: {np.unravel_index(near_ir.argmax(),near_ir.shape)}")
    # print(f"min near_ir: {near_ir.min()} at: {np.unravel_index(near_ir.argmin(),near_ir.shape)}")
    ref = client.destagger(source.metadata,scan.field(client.ChanField.REFLECTIVITY))
    if LIMITS_IRR["reflectivity"] not in [0,None]:
        ref = np.divide(ref,LIMITS_IRR["reflectivity"]) if LIMITS_IRR["reflectivity"] else ref
        ref = np.where(ref<1,ref,1)
    # print(f"max ref: {ref.max()} at: {np.unravel_index(ref.argmax(),ref.shape)}")
    # print(f"min ref: {ref.min()} at: {np.unravel_index(ref.argmin(),ref.shape)}")
    range_ous = client.destagger(source.metadata,scan.field(client.ChanField.RANGE))
    if LIMITS_IRR["range"] not in [0,None]:
        range_ous = np.divide(range_ous,LIMITS_IRR["range"])
        range_ous = np.where(range_ous<1,range_ous,1)
    
    # print(f"max range: {range_ous.max()} at: {np.unravel_index(range_ous.argmax(),range_ous.shape)}")
    # print(f"min range: {range_ous.min()} at: {np.unravel_index(range_ous.argmin(),range_ous.shape)}")
    stack = np.stack([near_ir,ref,range_ous],axis=2).astype(np.float32)
    return stack
def signal_ref_range(source,scan):
    
    signal = client.destagger(source.metadata,scan.field(client.ChanField.SIGNAL))
    if LIMITS_SRR["signal"] not in [0,None]:
        signal = np.divide(signal,LIMITS_IRR["signal"], dtype=np.float32)
        signal = np.where(signal<1,signal,1)
    # print(f"max near_ir: {near_ir.max()} at: {np.unravel_index(near_ir.argmax(),near_ir.shape)}")
    # print(f"min near_ir: {near_ir.min()} at: {np.unravel_index(near_ir.argmin(),near_ir.shape)}")
    ref = client.destagger(source.metadata,scan.field(client.ChanField.REFLECTIVITY))
    if LIMITS_SRR["reflectivity"] not in [0,None]:
        ref = np.divide(ref,LIMITS_SRR["reflectivity"]) if LIMITS_SRR["reflectivity"] else ref
        ref = np.where(ref<1,ref,1)
    # print(f"max ref: {ref.max()} at: {np.unravel_index(ref.argmax(),ref.shape)}")
    # print(f"min ref: {ref.min()} at: {np.unravel_index(ref.argmin(),ref.shape)}")
    range_ous = client.destagger(source.metadata,scan.field(client.ChanField.RANGE))
    if LIMITS_SRR["range"] not in [0,None]:
        range_ous = np.divide(range_ous,LIMITS_SRR["range"])
        range_ous = np.where(range_ous<1,range_ous,1)
    
    # print(f"max range: {range_ous.max()} at: {np.unravel_index(range_ous.argmax(),range_ous.shape)}")
    # print(f"min range: {range_ous.min()} at: {np.unravel_index(range_ous.argmin(),range_ous.shape)}")
    stack = np.stack([signal,ref,range_ous],axis=2).astype(np.float32)
    return stack
def record_cv2_images(args):
    """
    Stream Live from sensor belonging to hostname, given a specified config.
    @param config: SensorConfig object
    @param hostname: string
    @param lidar_port: int (default 7502)
    @param imu_port: int (default 7503)
    """
    lidar_port = args.lidar_port
    imu_port = args.imu_port
    hostname = args.hostname
    host_ip = args.host_ip
    frames_to_record = args.frames_to_record
    #plt.ion()
    if host_ip is not None:
        config = sensor_config(hostname=host_ip,lidar_port=lidar_port,imu_port=imu_port)
        hostname = host_ip
    elif hostname is not None:
        config = sensor_config(hostname=hostname,lidar_port=lidar_port,imu_port=imu_port)
    # create a stream object
    print(lidar_port,imu_port,hostname)
    save_path_IRR = f"../lidarImages/Ir-Ref-Range/{args.scene_name}"
    save_path_SRR = f"../lidarImages/Signal-Ref-Range/{args.scene_name}"
    if not os.path.exists(save_path_IRR):
        os.makedirs(save_path_IRR)
    else:
        print(f"{save_path_IRR} already exists")
        save_path_IRR = f"{save_path_IRR}{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        print(f"Saving to {save_path_IRR}")
        os.mkdir(save_path_IRR)
    if not os.path.exists(save_path_SRR):
        os.makedirs(save_path_SRR)
    else:
        print(f"{save_path_SRR} already exists")
        save_path_SRR = f"{save_path_SRR}{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        print(f"Saving to {save_path_SRR}")
        os.mkdir(save_path_SRR)
    print(f"Start Lidar Recording: \nInterval {args.time_to_wait} seconds,\nFrames to record {args.frames_to_record}")
    with closing(client.Scans.stream(hostname, lidar_port,
                                    complete=True)) as stream:
        first_scan = next(iter(stream))
        xyz = get_xyz(stream,first_scan)
        signal = get_signal_reflection(stream,first_scan)
        xyzr = convert_to_xyzr(xyz,signal)
        comp_xyzr = compress_mid_dim(xyzr)
        comp_xyzr = trim_xyzr(comp_xyzr,[25,25,25])
        wait_for_input = args.wait_for_input
        #vis, geo = initialize_o3d_plot(comp_xyzr)
        limits = LIMITS_IRR
        prev_time = time.monotonic()
        # start the stream
        i = 0
        imgsz = [1280,640]
        ttw = args.time_to_wait if args.time_to_wait is not None else 0
        for scan in stream:
            if wait_for_input:
                input("Press Enter to Record...")
            # uncomment if you'd like to see frame id printed
            # print("frame id: {} ".format(scan.frame_id))
            current_time = time.monotonic()
            if current_time - prev_time > ttw or wait_for_input:
                prev_time = current_time
                signal = get_signal_reflection(stream,scan)
                xyz = get_xyz(stream,scan)
                xyzr = convert_to_xyzr(xyz,signal)
                comp_xyzr = compress_mid_dim(xyzr)
                comp_xyzr = trim_xyzr(comp_xyzr,[limits["range"]/1000,limits["range"]/1000,limits["range"]/1000]) # Convert to meters
                #update_open3d_live(geo,comp_xyzr,vis)
                img_IRR = ir_ref_range(stream,scan)
                img_SRR = signal_ref_range(stream,scan)
                #print(f"Average: \nIR {np.mean(img[:,:,0])} \nRange {np.mean(img[:,:,1])} \nReflectivity {np.mean(img[:,:,2])}")
                filename = f"{save_path}/image_{i}.jpg"
                cv2.imwrite(filename,cv2.cvtColor(img_SRR*255,cv2.COLOR_RGB2BGR))
                cv2.imwrite(filename,cv2.cvtColor(img_IRR*255,cv2.COLOR_RGB2BGR))

                cv2.imshow("IRR", cv2.resize(cv2.cvtColor(copy(img_IRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.imshow("SRR", cv2.resize(cv2.cvtColor(copy(img_SRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.waitKey(1)
                i += 1
            else:
                img_IRR = ir_ref_range(stream,scan)
                img_SRR = signal_ref_range(stream,scan)
                #print(f"Average: \nIR {np.mean(img[:,:,0])} \nRange {np.mean(img[:,:,1])} \nReflectivity {np.mean(img[:,:,2])}")
                cv2.imshow("IRR", cv2.resize(cv2.cvtColor(copy(img_IRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.imshow("SRR", cv2.resize(cv2.cvtColor(copy(img_IRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.waitKey(1)

            if i>=frames_to_record:
                break

def count_down():
    start = time.monotonic()
    for i in range(3):
        print(3-i)
        time.sleep(1)
def record_cv2_images_dual(args):
    """
    Stream Live from sensor belonging to hostname, given a specified config.
    @param config: SensorConfig object
    @param hostname: string
    @param lidar_port: int (default 7502)
    @param imu_port: int (default 7503)
    """
    parent_path = "../lidarImages/New-Data"
    lidar_port = args.lidar_port
    imu_port = args.imu_port
    hostname = args.hostname
    host_ip = args.host_ip
    frames_to_record = args.frames_to_record
    #plt.ion()
    if host_ip is not None:
        config = sensor_config(hostname=host_ip,lidar_port=lidar_port,imu_port=imu_port)
        hostname = host_ip
    elif hostname is not None:
        config = sensor_config(hostname=hostname,lidar_port=lidar_port,imu_port=imu_port)
    # create a stream object
    print(lidar_port,imu_port,hostname)
    save_path_IRR = f"{parent_path}/Ir-Ref-Range/{args.scene_name}"
    save_path_SRR = f"{parent_path}/Signal-Ref-Range/{args.scene_name}"
    if not os.path.exists(save_path_IRR):
        os.makedirs(save_path_IRR)
    else:
        print(f"{save_path_IRR} already exists")
        save_path_IRR = f"{save_path_IRR}{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        print(f"Saving to {save_path_IRR}")
        os.mkdir(save_path_IRR)
    if not os.path.exists(save_path_SRR):
        os.makedirs(save_path_SRR)
    else:
        print(f"{save_path_SRR} already exists")
        save_path_SRR = f"{save_path_SRR}{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        print(f"Saving to {save_path_SRR}")
        os.mkdir(save_path_SRR)
    
    print(f"Start Lidar Recording: \nInterval {args.time_to_wait} seconds,\nFrames to record {args.frames_to_record}")
    with closing(client.Scans.stream(hostname, lidar_port,
                                    complete=True)) as stream:
        first_scan = next(iter(stream))
        xyz = get_xyz(stream,first_scan)
        signal = get_signal_reflection(stream,first_scan)
        xyzr = convert_to_xyzr(xyz,signal)
        comp_xyzr = compress_mid_dim(xyzr)
        comp_xyzr = trim_xyzr(comp_xyzr,[25,25,25])
        wait_for_input = args.wait_for_input
        #vis, geo = initialize_o3d_plot(comp_xyzr)
        #LIMITS_IRR = {"ir":6000,"reflectivity": 255, "range":25000}
        # start the stream
        i = 0
        prev_time = 0
        ttw = args.time_to_wait if args.time_to_wait is not None else 0
        imgsz = [1280,640]
        if not wait_for_input:
            count_down()
        for scan in stream:
        
            if wait_for_input:
                input("Press Enter to Record...")
            # uncomment if you'd like to see frame id printed
            # print("frame id: {} ".format(scan.frame_id))
            current_time = time.monotonic()
            if current_time - prev_time > ttw or wait_for_input:
                prev_time = time.monotonic()
                signal = get_signal_reflection(stream,scan)
                xyz = get_xyz(stream,scan)
                xyzr = convert_to_xyzr(xyz,signal)
                comp_xyzr = compress_mid_dim(xyzr)
                comp_xyzr = trim_xyzr(comp_xyzr,[LIMITS_IRR["range"]/1000,LIMITS_IRR["range"]/1000,LIMITS_IRR["range"]/1000]) # Convert to meters
                #update_open3d_live(geo,comp_xyzr,vis)
                img_IRR = ir_ref_range(stream,scan)
                img_SRR = signal_ref_range(stream,scan)
                #print(f"Average: \nIR {np.mean(img[:,:,0])} \nRange {np.mean(img[:,:,1])} \nReflectivity {np.mean(img[:,:,2])}")
                filename_IRR = f"{save_path_IRR}/image_{i}.jpg"
                filename_SRR = f"{save_path_SRR}/image_{i}.jpg"
                cv2.imwrite(filename_SRR,cv2.resize(cv2.cvtColor(copy(img_SRR)*255,cv2.COLOR_RGB2BGR),imgsz))
                cv2.imwrite(filename_IRR,cv2.resize(cv2.cvtColor(copy(img_IRR)*255,cv2.COLOR_RGB2BGR),imgsz))

                cv2.imshow("IRR", cv2.resize(cv2.cvtColor(copy(img_IRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.imshow("SRR", cv2.resize(cv2.cvtColor(copy(img_SRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.waitKey(1)
                i += 1
                print(f"Recorded frame {i}/{frames_to_record}")
            else:
                img_IRR = ir_ref_range(stream,scan)
                img_SRR = signal_ref_range(stream,scan)
                #print(f"Average: \nIR {np.mean(img[:,:,0])} \nRange {np.mean(img[:,:,1])} \nReflectivity {np.mean(img[:,:,2])}")
                cv2.imshow("IRR", cv2.resize(cv2.cvtColor(copy(img_IRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.imshow("SRR", cv2.resize(cv2.cvtColor(copy(img_SRR),cv2.COLOR_RGB2BGR),imgsz))
                cv2.waitKey(1)
            if i>=frames_to_record:
                break
import argparse  
import sys         
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser') 
    #parser.add_argument('--hostname', type=str, default=None, help='hostname')
    parser.add_argument('--hostname', nargs="+", default=None, help='hostname/(s)')
    #parser.add_argument('--lidar_port', type=int, default=7502, help='lidar port')
    parser.add_argument('--lidar_port', nargs="+", default=[7502], help='lidar port/(s)')
    #parser.add_argument('--imu_port', type=int, default=7503, help='imu port')
    parser.add_argument('--imu_port', nargs="+", default=[7503], help='imu port/(s)')
    #parser.add_argument('--host_ip', type=str, default="192.168.200.78", help='ip address of host')
    parser.add_argument('--host_ip', nargs="+", default=["192.168.200.78"],help='ip address of host')
    parser.add_argument('--phase_locking', nargs="+", default=[0], help='where to phase lock each sensor')
    parser.add_argument('--scene_name', type=str, default='scene_1', help='scene name')
    parser.add_argument('--frames_to_record', type=int, default=25, help='frames to record')
    parser.add_argument('--stream_time', type=int, default=100, help='time to stream live')
    parser.add_argument('--time_to_wait', type=float, default=3, help='time to wait between images')
    parser.add_argument('--relative_position',nargs="+",action='append',help='relative position of lidar sensor 2 in scene')
    if sys.version_info >= (3,9):
        parser.add_argument('--wait_for_input', action=argparse.BooleanOptionalAction)
    else:
        parser.add_argument('--wait_for_input', action='store_true')
        parser.add_argument('--no-wait_for_input', action='store_false')
    args = parser.parse_args()
    if args.hostname is None:
        args.hostname = args.host_ip
    if len(args.hostname)==1:
        args.hostname = args.hostname[0]
        args.lidar_port = int(args.lidar_port[0])
        args.imu_port = int(args.imu_port[0])
        args.host_ip = args.host_ip[0]

    return args


if __name__ == "__main__":
    #stream_from_multiple(parse_config()) # py utils_ouster.py --lidar_port 7502 7504 --imu_port 7503 7505  --host_ip 192.168.200.78 192.168.200.79 --stream_time 500 --relative_position 0 0 0 --relative_position -4.40 0 0
    record_cv2_images_dual(parse_config()) 
    #python3 utils_ouster.py --scene_name "Testing" --no-wait_for_input --time_to_wait 4 --frames_to_record 10
    #filename,_ = record_lidar_seq(seq_length=5)
    #filename = "../lidar_scans/Huuuman"
    #print("filename: {}".format(filename))
    #play_back_recording(filename,trim=[10,10,10])
    #scan,source = get_single_example()
    #xyz = get_xyz(source,scan)
    #signal = get_signal_reflection(scan,source)
    #xyzr = convert_to_xyzr(xyz,signal)
    #plot_open3d_pc(compress_mid_dim(xyzr))
