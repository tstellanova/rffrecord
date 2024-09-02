#!/usr/bin/env python
"""
Record SAR data at full bandwidth
using HackRF SDR, and output in SigMF format
"""
from subprocess import Popen, PIPE, STDOUT
import re
import argparse
import os
import json
from datetime import datetime, timezone

import numpy as np
import sigmf
from sigmf import SigMFFile


def main():
    parser = argparse.ArgumentParser(description='Grab some SAR data using hackrf_transfer')
    parser.add_argument('--duration', '-d',  type=int, default=15,
                        help='Duration to capture, in seconds')
    parser.add_argument('--serial_num', '-sn',   default=None,
                        help='Specific HackRF serial number to use')
    parser.add_argument('--center_freq_mhz', '-fc', dest='fc_mhz', type=float, default=5405.5000,
                        help='Center frequency to record, in MHz')
    parser.add_argument("--out_path",dest='out_path',default='./data/',
                        help="Directory path to place output files" )

    args = parser.parse_args()
    duration_seconds = args.duration
    specific_hrf_sn = args.serial_num
    freq_ctr_mhz = args.fc_mhz
    out_path = args.out_path

    if not os.path.isdir(out_path):
        print(f"out_path {out_path} does not exist")
        return -1

    sampling_bw_mhz = 20.0 # full bandwidth of HackRF
    true_bb_bw_mhz = sampling_bw_mhz

    print(f"Ctr Freq: {freq_ctr_mhz} MHz | BW : {sampling_bw_mhz} MHz | duration: {duration_seconds} s")

    sample_rate_hz = int(sampling_bw_mhz * 1E6)
    ctr_freq_hz = int(freq_ctr_mhz * 1E6)
    # TODO determine whether bb filter bw > sampling bw is more optimal for SAR
    baseband_filter_bw_hz = sample_rate_hz

    # These are based on testing with some antenna and LNA combos -- YMMV
    if_lna_gain_db, baseband_gain_db = 40, 20

    n_samples = int(duration_seconds * sample_rate_hz)

    # for SigMF annotation: calculate the band occupied by the signal
    half_baseband_bandwidth = int(baseband_filter_bw_hz / 2)
    freq_lower_edge = int(ctr_freq_hz - half_baseband_bandwidth)
    freq_upper_edge = int(ctr_freq_hz + half_baseband_bandwidth)

    # figure out where to put the output files automatically
    file_number = 1
    path_stem = f'{out_path}hrf_sar_{ctr_freq_hz}_{duration_seconds}s'
    data_out_path = f'{path_stem}_{file_number:04d}.sigmf-data'
    while os.path.isfile(data_out_path):
        file_number += 1
        data_out_path = f'{path_stem}_{file_number:04d}.sigmf-data'
    meta_out_path = f'{path_stem}_{file_number:04d}.sigmf-meta'

    # assumes that HackrF software version is new enough to support `-B` power reporting flag
    opt_str = f"-f {ctr_freq_hz} -a 1 -l {if_lna_gain_db} -g {baseband_gain_db} -b {baseband_filter_bw_hz} -s {sample_rate_hz} -n {n_samples}  -B -r {data_out_path}"
    if specific_hrf_sn is None:
        cmd_str = f"hackrf_transfer {opt_str}"
    else:
        cmd_str = f"hackrf_transfer -d {specific_hrf_sn} {opt_str}"

    print(f"START:\n{cmd_str} ")

    # Regex to match and extract numeric values
    regex = r"[-+]?\d*\.\d+|\d+"

    total_power = float(0)
    step_count = 0
    line_count = 0
    capture_start_utc = None

    with (Popen([cmd_str], stdout=PIPE, stderr=STDOUT, text=True, shell=True) as proc):
        for line in proc.stdout:
            if line_count > 6: # skip command startup lines
                if capture_start_utc is None:
                    capture_start_utc = datetime.utcnow().isoformat()+'Z'

                numeric_values = re.findall(regex, line)
                if numeric_values is not None and len(numeric_values) == 7:
                    # 8.1 MiB / 1.000 sec =  8.1 MiB/second, average power -2.0 dBfs, 14272 bytes free in buffer, 0 overruns, longest 0 bytes
                    # ['8.1', '1.000', '8.1', '-2.0', '14272', '0', '0']
                    print(numeric_values)
                    step_power = numeric_values[3]
                    total_power += float(step_power)
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
        print(f"avg_power: {avg_power:02.3f} (dBFS)")


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
        'stellanovat:LNA_pwr': 'USB-C',
    },
    "captures": [
        {
            SigMFFile.START_INDEX_KEY: 0,
            SigMFFile.FREQUENCY_KEY: int(f'{ctr_freq_hz}'), 
            SigMFFile.DATETIME_KEY: f'{capture_start_utc}',
            'stellanovat:if_gain_db': int(f'{if_lna_gain_db}'),
            'stellanovat:bb_gain_db': int(f'{baseband_gain_db}'),
            'stellanovat:sdr_rx_amp_enabled': 0,
            "stellanovat:recorder_command": f'{cmd_str}',
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

    meta_json = json.dumps(meta_info_dict, indent=2)

    with open(meta_out_path, "w") as meta_outfile:
        meta_outfile.write(meta_json)

    print(f"wrote {meta_out_path}")



if __name__ == "__main__":
    main()
