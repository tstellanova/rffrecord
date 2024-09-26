#!/usr/bin/env python
"""
Continuously SAR data at full bandwidth
using HackRF SDR, and output in SigMF format
"""
from subprocess import Popen, PIPE, STDOUT
import re
import argparse
import os
import shutil
import json
from datetime import datetime, timezone

import numpy as np
import sigmf
from sigmf import SigMFFile


def capture_one_data_segment(cmd_str_stem=None, data_out_path=None):
    # assumes that HackrF software version supports `-B` power reporting flag
    cmd_str = f"{cmd_str_stem} -r {data_out_path}"

    print(f"START:\n{cmd_str} ")

    # Regex to match and extract numeric values
    regex = r"[-+]?\d*\.\d+|\d+"

    total_power = float(0)
    avg_power = float(0)
    max_power = float(-200)
    step_count = 0
    line_count = 0
    capture_start_utc = None

    with (Popen([cmd_str], stdout=PIPE, stderr=STDOUT, text=True, shell=True) as proc):
        for line in proc.stdout:
            if line_count > 6:  # skip command startup lines
                if capture_start_utc is None:
                    capture_start_utc = datetime.utcnow().isoformat() + 'Z'

                numeric_values = re.findall(regex, line)
                if numeric_values is not None and len(numeric_values) == 7:
                    # 8.1 MiB / 1.000 sec =  8.1 MiB/second, average power -2.0 dBfs, 14272 bytes free in buffer, 0 overruns, longest 0 bytes
                    # ['8.1', '1.000', '8.1', '-2.0', '14272', '0', '0']
                    print(numeric_values)
                    step_power = float(numeric_values[3])
                    if step_power > max_power:
                        max_power = step_power
                    total_power += step_power
                    step_count += 1
                else:
                    # read all the stdout until finished, else data out files are not flushed
                    continue
            line_count += 1

    rc = proc.returncode
    if 0 != rc:
        print(f"hackrf_transfer failed with result code: {rc}")
    else:
        avg_power = total_power / float(step_count)
        print(f"max_power: {max_power:0.3f} avg_power: {avg_power:0.3f} (dBFS)")

    return max_power, avg_power


def main():
    parser = argparse.ArgumentParser(description="""
        Grab some SAR data using hackrf_transfer""")
    parser.add_argument('--duration', '-d', type=int, default=15,
                        help='Duration to capture, in seconds')
    parser.add_argument('--serial_num', '-sn', default=None,
                        help='Specific HackRF serial number to use')
    parser.add_argument('--center_freq_mhz', '-fc', dest='fc_mhz', type=float, default=5405.5000,
                        help='Center frequency to record, in MHz')
    parser.add_argument("--tmp_path", dest='tmp_path', default=None,
                        help="Directory path to place temporary files (e.g. a ramdisk)")
    parser.add_argument("--out_path", dest='out_path', default='../../baseband/sar-recordings/',
                        help="Directory path to place output files")
    parser.add_argument('--squelch_dbfs', dest='squelch_dbfs', type=float, default=-29.0,
                        help="In nonstop mode, the minimum recorded power to keep")
    parser.add_argument('--delta_dbfs', dest='min_peak_gap_dbfs', type=float, default=1.1,
                        help="Minimum peak dbFS above average for us to keep a recording")
    args = parser.parse_args()
    duration_seconds = args.duration
    specific_hrf_sn = args.serial_num
    freq_ctr_mhz = args.fc_mhz
    peak_squelch_dbfs = args.squelch_dbfs
    min_peak_gap_dbfs = args.min_peak_gap_dbfs
    out_path = args.out_path
    tmp_path = out_path
    if args.tmp_path is not None:
        tmp_path = args.tmp_path

    if not os.path.isdir(out_path):
        print(f"out_path {out_path} does not exist")
        return -1

    if not os.path.isdir(tmp_path):
        print(f"tmp_path {tmp_path} does not exist")
        return -1

    sampling_bw_mhz = 20.0  # full bandwidth of HackRF

    print(f"Ctr Freq: {freq_ctr_mhz} MHz | BW : {sampling_bw_mhz} MHz | duration: {duration_seconds} s")
    print(f"Squelch: {peak_squelch_dbfs} dbFS | Peak Delta : {min_peak_gap_dbfs} dbFS")

    sample_rate_hz = int(sampling_bw_mhz * 1E6)
    ctr_freq_hz = int(freq_ctr_mhz * 1E6)
    # TODO determine whether bb filter bw > sampling bw is more optimal for SAR
    baseband_filter_bw_hz = sample_rate_hz

    # These are based on testing with some antenna and LNA combos -- YMMV
    if_lna_gain_db, baseband_gain_db = 40, 24

    n_samples = int(duration_seconds * sample_rate_hz)

    # for SigMF annotation: calculate the band occupied by the signal
    half_baseband_bandwidth = int(baseband_filter_bw_hz / 2)
    freq_lower_edge = int(ctr_freq_hz - half_baseband_bandwidth)
    freq_upper_edge = int(ctr_freq_hz + half_baseband_bandwidth)

    base_filename_stem = f'hrf_sar_{int(freq_ctr_mhz)}_{duration_seconds}s'

    # assumes that HackrF software version supports `-B` power reporting flag
    opt_str = f"-f {ctr_freq_hz} -a 1 -l {if_lna_gain_db} -g {baseband_gain_db} -b {baseband_filter_bw_hz} -s {sample_rate_hz} -n {n_samples}  -B "
    if specific_hrf_sn is None:
        cmd_str_stem = f"hackrf_transfer {opt_str}"
    else:
        cmd_str_stem = f"hackrf_transfer -d {specific_hrf_sn} {opt_str}"

    basic_capture_start_utc = datetime.utcnow().isoformat() + 'Z'

    # TODO look at using the SigMFFile object, directly, instead
    meta_info_dict = {
        "global": {
            SigMFFile.DATATYPE_KEY: 'ci8',
            SigMFFile.SAMPLE_RATE_KEY: int(f'{sample_rate_hz}'),
            SigMFFile.HW_KEY: "HackRF, LNA, antenna",
            SigMFFile.AUTHOR_KEY: 'Todd Stellanova',
            SigMFFile.VERSION_KEY: f'{sigmf.__version__}',
            SigMFFile.DESCRIPTION_KEY: f'SAR recorded using hackrf_transfer',
            SigMFFile.RECORDER_KEY: 'hackrf_transfer',
            'antenna:type': 'Wideband',
            'stellanovat:sdr': 'HackRF',
            'stellanovat:sdr_sn': f'{specific_hrf_sn}',
            'stellanovat:LNA': '6GHz 20dB',
            'stellanovat:LNA_pwr': 'bias-tee',
        },
        "captures": [
            {
                SigMFFile.START_INDEX_KEY: 0,
                SigMFFile.FREQUENCY_KEY: int(f'{ctr_freq_hz}'),
                SigMFFile.DATETIME_KEY: f'{basic_capture_start_utc}',  # replace later
                'stellanovat:if_gain_db': int(f'{if_lna_gain_db}'),
                'stellanovat:bb_gain_db': int(f'{baseband_gain_db}'),
                'stellanovat:sdr_rx_amp_enabled': 1,
                "stellanovat:recorder_command": f'{cmd_str_stem}',
                "stellanovat:max_power_dbfs": 0
            }
        ],
        "annotations": [
            {
                SigMFFile.START_INDEX_KEY: 0,
                SigMFFile.LENGTH_INDEX_KEY: int(f'{n_samples}'),
                SigMFFile.FHI_KEY: int(f'{freq_upper_edge}'),
                SigMFFile.FLO_KEY: int(f'{freq_lower_edge}'),
                SigMFFile.LABEL_KEY: f'SAR',
            }
        ]
    }

    while True:
        le_datetime = datetime.utcnow()
        seg_start_time_utc = le_datetime.isoformat() + 'Z'
        compact_datetime_str = le_datetime.isoformat(sep='_', timespec='seconds') + 'Z'
        # convert eg: '2024-09-15_04:28:13Z'  into: `20240915_042813Z`
        more_compact_datetimestr = re.sub('\-|\:', '', compact_datetime_str)
        full_filename_stem = f'{base_filename_stem}_{more_compact_datetimestr}'
        # first we will write data to a temporary complex (I/Q) signed byte file
        tmp_data_file_path = f'{tmp_path}{full_filename_stem}.cs8'
        max_power, avg_power = capture_one_data_segment(cmd_str_stem, tmp_data_file_path)
        keep_segment = False
        power_delta = max_power - avg_power  # for a legit signal, max power should well exceed average
        if (power_delta >= min_peak_gap_dbfs) or (max_power > peak_squelch_dbfs):
            keep_segment = True
        print(
            f"check (peak > squelch): {max_power:0.2f} > {peak_squelch_dbfs} or (peak - avg) {power_delta:0.2f} > {min_peak_gap_dbfs} ")
        if keep_segment:
            # move the tmp data file to a more persistent location
            solid_data_file_path = f'{out_path}{full_filename_stem}.sigmf-data'
            print(f"moving {tmp_data_file_path} to {solid_data_file_path} ...")
            if tmp_path == out_path:
                os.rename(tmp_data_file_path, solid_data_file_path)
            else:
                # this method takes care of deleting the tmp file
                shutil.move(tmp_data_file_path, solid_data_file_path)

            # create a meta file for the data
            meta_info_dict["captures"][0][SigMFFile.DATETIME_KEY] = seg_start_time_utc
            meta_info_dict["captures"][0]["stellanovat:max_power_dbfs"] = max_power
            meta_out_path = f'{out_path}{full_filename_stem}.sigmf-meta'
            meta_json = json.dumps(meta_info_dict, indent=2)

            with open(meta_out_path, "w") as meta_outfile:
                meta_outfile.write(meta_json)
            print(f"wrote:\n{meta_out_path}")
        else:
            # remove the data file that did not meet squelch standard
            print(f"deleting {tmp_data_file_path} ...")
            os.remove(tmp_data_file_path)


if __name__ == "__main__":
    main()
