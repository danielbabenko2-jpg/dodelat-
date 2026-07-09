#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import sys
sys.path.append("../")
from common.bus_call import bus_call
from common.platform_info import PlatformInfo
import pyds
import platform
import math
import time
from ctypes import *
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib
import configparser
import datetime

import argparse
import ctypes
import cupy as cp
import cv2
import numpy as np
import os
from pathlib import Path

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]

# Глобальные переменные для background subtractor и видео
frame_cnt = 0
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=36,
    detectShadows=False
)

# Переменные для сохранения видео
video_writer = None
video_frames = []
VIDEO_FRAMES_LIMIT = 100
video_counter = 0
output_dir = "output_videos"

# Создаем директорию для видео, если её нет
Path(output_dir).mkdir(parents=True, exist_ok=True)

def on_new_sample(appsink, user_data):
    # 1. Получить буфер из appsink
    sample = appsink.emit("pull-sample")
    if not sample:
        print("ERROR: Failed to pull sample")
        return Gst.FlowReturn.ERROR
    
    gst_buffer = sample.get_buffer()
    if not gst_buffer:
        print("ERROR: Failed to get buffer")
        return Gst.FlowReturn.ERROR
            
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        print("ERROR: No batch meta found")
        return Gst.FlowReturn.ERROR
    
    l_frame = batch_meta.frame_meta_list
    frames_processed = 0
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_index = frame_meta.batch_id
        global frame_cnt
        frame_cnt += 1
        
        # Получаем данные кадра с GPU
        data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(
            hash(gst_buffer), frame_index
        )
        
        if not dataptr:
            print(f"ERROR: Failed to get surface for frame {frame_index}")
            l_frame = l_frame.next
            continue
        
        # Конвертируем данные в numpy array
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
        unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner=None) 
        memptr = cp.cuda.MemoryPointer(unownedmem, 0)

        # Создание массива CuPy (все еще на GPU)
        n_frame_gpu = cp.ndarray(
            shape=shape, 
            dtype=data_type, 
            memptr=memptr, 
            strides=strides, 
            order='C'
        )
        n_frame_cpu = cp.asnumpy(n_frame_gpu)
        
        # Применяем background subtractor (используем BGR напрямую)
        fg_mask = bg_subtractor.apply(n_frame_cpu)
        
        # Морфологическая обработка
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создаем черно-белое изображение с рамками
        frame_with_boxes = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        # Рисуем рамки
        min_area = 500
        objects_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_with_boxes, f'Obj {int(area)}', 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                objects_count += 1
        
        # Добавляем информацию
        cv2.putText(frame_with_boxes, f'Frame: {frame_cnt}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_with_boxes, f'Objects: {objects_count}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if ts_from_rtsp:
            ts = frame_meta.ntp_timestamp / 1000000000
            time_str = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame_with_boxes, f'Time: {time_str}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Конвертируем в байты
        frame_bytes = frame_with_boxes.tobytes()
   
        
        # Создаем новый буфер для appsrc
        new_buffer = Gst.Buffer.new_allocate(None, len(frame_bytes), None)
        if not new_buffer:
            print("ERROR: Failed to allocate new buffer")
            l_frame = l_frame.next
            continue
        
        new_buffer.fill(0, frame_bytes)
        new_buffer.pts = gst_buffer.pts
        new_buffer.dts = gst_buffer.dts
        new_buffer.duration = gst_buffer.duration
        
        
        # Отправляем в appsrc
        appsrc = user_data
        
        
        ret = appsrc.emit("push-buffer", new_buffer)
        
        if ret != Gst.FlowReturn.OK:
            print(f"ERROR: Failed to push buffer: {ret}")
            return Gst.FlowReturn.ERROR
        
        frames_processed += 1
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    if frames_processed == 0:
        return Gst.FlowReturn.ERROR
    return Gst.FlowReturn.OK

def pgie_src_pad_buffer_probe(pad, info, u_data):
    # global frame_cnt, video_counter, video_frames
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        # frame_cnt += 1
        
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        
        frame_number = frame_meta.frame_num
        print(f"Frame Number={frame_number}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# Остальные функции (cb_newpad, decodebin_child_added, create_source_bin, main, parse_args)
# остаются без изменений

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if ts_from_rtsp:
        if name.find("source") != -1:
            pyds.configure_source_for_ntp_sync(hash(Object))


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target(
            "src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    number_sources = len(args)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    print("Creating Pgie \n ")
    if gie=="nvinfer":
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating tiler \n ")

    nvvideoconv_opencv = Gst.ElementFactory.make('nvvideoconvert','opencv_convert')
    if not nvvideoconv_opencv:
        sys.stderr.write(" Unable to create opencv convert")

    caps_opencv = Gst.ElementFactory.make("capsfilter", "filter_opencv")
    caps_opencv.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    caps_pre_tiler = Gst.ElementFactory.make("capsfilter", "filter_pre_tiler")
    caps_pre_tiler.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )

    nvconv_pre_tiler = Gst.ElementFactory.make("nvvideoconvert", "convertor_pre_tiler")
    if not nvconv_pre_tiler:
        sys.stderr.write(" Unable to create nvvidconv_pre_tiler \n")

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvvidconv_postosd = Gst.ElementFactory.make(
        "nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    #CREATE TEE
    tee = Gst.ElementFactory.make("tee","tee")
    if not tee:
        sys.stderr.write(" Unable to create tee")

    #CREATE APPSINK
    appsink = Gst.ElementFactory.make("appsink","appsink")
    if not appsink:
        sys.stderr.write(" Unable to create appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)
    appsink.set_property("async", False)
    appsink.set_property("drop", True)
    appsink.set_property("max-buffers", 2)
    
    nvvidconv_post_appsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_post_appsrc")
    if not nvvidconv_post_appsrc:
        sys.stderr.write(" Unable to create nvvideoconv_post_appsrc")

    appsrc = Gst.ElementFactory.make("appsrc","appsrc")
    if not appsrc:
        sys.stderr.write(" Unable to create appsrc")
    # Настройка caps для appsrc (должны совпадать с тем, что выдает appsink)
    appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw, format=BGR, width=1920, height=1080, framerate=10/1"))
    appsrc.set_property("format", Gst.Format.TIME)
    appsrc.set_property("is-live", True)
    appsrc.set_property("do-timestamp", True)
    appsrc.set_property("block", True)  # Блокировать, если очередь полна

    appsink.connect("new-sample", on_new_sample, appsrc)  # Ваш колбэк


    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    )
    
    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        encoder_post_appsrc = Gst.ElementFactory.make("nvv4l2h264enc", "encoder_2")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        encoder_post_appsrc = Gst.ElementFactory.make("nvv4l2h265enc", "encoder_2")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    if platform_info.is_integrated_gpu():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        #encoder.set_property("bufapi-version", 1)
    encoder_post_appsrc.set_property("bitrate", bitrate)
    if platform_info.is_integrated_gpu():
        encoder_post_appsrc.set_property("preset-level", 1)
        encoder_post_appsrc.set_property("insert-sps-pps", 1)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        rtppay.set_property("config_interval", 1)
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")

    if codec == "H264":
        rtppay_2 = Gst.ElementFactory.make("rtph264pay", "rtppay_2")
        rtppay_2.set_property("config_interval", 1)
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay_2 = Gst.ElementFactory.make("rtph265pay", "rtppay_2")
        print("Creating H265 rtppay")
    if not rtppay_2:
        sys.stderr.write(" Unable to create rtppay")

    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")

    updsink_port_num_2 = 5500
    sink_2 = Gst.ElementFactory.make("udpsink", "udpsink_2")
    if not sink_2:
        sys.stderr.write(" Unable to create udpsink")

    queue_bg = Gst.ElementFactory.make("queue", "queue_bg")
    if not queue_bg:
        sys.stderr.write(" Unable to create queue_bg")
        return -1

    # Настройка queue для буферизации
    queue_bg.set_property("leaky", 2)  # 2 = downstream (сбрасывает старые кадры если переполнено)
    queue_bg.set_property("max-size-buffers", 10)  # Максимум 10 буферов
    queue_bg.set_property("max-size-time", 0)      # 0 = не ограничивать по времени
    queue_bg.set_property("max-size-bytes", 0)     # 0 = не ограничивать по размеру

    sink.set_property("host", "224.224.255.255")
    sink.set_property("port", updsink_port_num)
    sink.set_property("async", False)
    sink.set_property("sync", 1)

    sink_2.set_property("host", "224.224.255.255")
    sink_2.set_property("port", updsink_port_num_2)
    sink_2.set_property("async", False)
    sink_2.set_property("sync", 1)

    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    
    if ts_from_rtsp:
        streammux.set_property("attach-sys-ts", 0)

    if gie=="nvinfer":
        pgie.set_property("config-file-path", "dstest1_pgie_config.txt")
    else:
        pgie.set_property("config-file-path", "dstest1_pgie_inferserver_config.txt")


    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)

    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)
    pipeline.add(sink_2)
    pipeline.add(rtppay_2)
    pipeline.add(caps_pre_tiler)
    pipeline.add(nvconv_pre_tiler)
    pipeline.add(tee)
    pipeline.add(appsink)
    pipeline.add(nvvideoconv_opencv)
    pipeline.add(caps_opencv)
    pipeline.add(appsrc)
    pipeline.add(nvvidconv_post_appsrc)
    pipeline.add(encoder_post_appsrc)
    pipeline.add(queue_bg)

    streammux.link(tee)
    tee.link(nvconv_pre_tiler)

    nvconv_pre_tiler.link(caps_pre_tiler)
    caps_pre_tiler.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tiler)
    tiler.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    tee.link(nvvideoconv_opencv)
    nvvideoconv_opencv.link(caps_opencv)
    caps_opencv.link(appsink)

    appsrc.link(nvvidconv_post_appsrc)
    nvvidconv_post_appsrc.link(encoder_post_appsrc)
    encoder_post_appsrc.link(rtppay_2)
    rtppay_2.link(queue_bg)  # <-- Добавляем queue
    queue_bg.link(sink_2)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_src_pad=pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print(
        "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
        % rtsp_port_num
    )

    rtsp_port_num_2 = 8555

    server_2 = GstRtspServer.RTSPServer.new()
    server_2.props.service = "%d" % rtsp_port_num_2
    server_2.attach(None)

    factory_2 = GstRtspServer.RTSPMediaFactory.new()
    factory_2.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num_2, codec)
    )
    factory_2.set_shared(True)
    server_2.get_mount_points().add_factory("/bg-test", factory)

    print(
        "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/bg-test ***\n\n"
        % rtsp_port_num_2
    )

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except BaseException:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                  help="Path to input H264 elementry stream", nargs="+", default=["a"], required=True)
    parser.add_argument("-g", "--gie", default="nvinfer",
                  help="choose GPU inference engine type nvinfer or nvinferserver , default=nvinfer", choices=['nvinfer','nvinferserver'])
    parser.add_argument("-c", "--codec", default="H264",
                  help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                  help="Set the encoding bitrate ", type=int)
    parser.add_argument("--rtsp-ts", action="store_true", default=False, dest='rtsp_ts', help="Attach NTP timestamp from RTSP source",
    )
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    global gie
    global ts_from_rtsp
    gie = args.gie
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    ts_from_rtsp = args.rtsp_ts
    return stream_path

if __name__ == '__main__':
    stream_path = parse_args()
    sys.exit(main(stream_path))
