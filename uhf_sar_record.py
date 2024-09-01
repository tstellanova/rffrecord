#!/usr/bin/env python
"""
Record SAR data from UHF
using HackRF SDR, and output in sigmf format
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
                        help='Specific HackRF serial number to use (default: None)')

    args = parser.parse_args()
    duration_seconds = args.duration
    specific_hrf_sn = args.serial_num

    print(f'duration {duration_seconds}')

    # 1227.6000, 11.000, 3.0, 2.5
    freq_ctr_mhz, full_bandwidth, sampling_bw_mhz, true_bb_bw_mhz = 430.5, 20.0, 20.0, 20.0
    print(f"ctr: {freq_ctr_mhz} bw: {sampling_bw_mhz}")

    if sampling_bw_mhz < 2.0:
        sampling_bw_mhz = 2.0
    sample_rate_hz = int(sampling_bw_mhz * 1E6)
    ctr_freq_hz = int(freq_ctr_mhz * 1E6)
    # 1.75/2.5/3.5/5/5.5/6/7/8/9/10/12/14/15/20/24/28MHz, default <= 0.75 * sample_rate_hz.
    bb_filter_mhz = true_bb_bw_mhz
    if bb_filter_mhz < 1.75:
        bb_filter_mhz = 1.75
    baseband_filter_bw_hz = int(bb_filter_mhz*1E6)

    if_lna_gain_db, baseband_gain_db = 32, 16

    n_samples = int(duration_seconds * sample_rate_hz)

    true_bb_bw_hz = int(true_bb_bw_mhz*1E6)
    print(f"true_bb_bw_hz: {true_bb_bw_hz}")
    half_baseband_bandwidth = int(true_bb_bw_hz / 2)
    freq_lower_edge = int(ctr_freq_hz - half_baseband_bandwidth)
    freq_upper_edge = int(ctr_freq_hz + half_baseband_bandwidth)
    # figure out where to put the output files automatically
    file_number = 1
    data_out_path = f'hrf_beale_sar_{duration_seconds}s_{file_number:04d}.sigmf-data'
    while os.path.isfile(data_out_path):
        file_number += 1
        data_out_path = f'hrf_beale_sar_{duration_seconds}s_{file_number:04d}.sigmf-data'
    meta_out_path = f'hrf_beale_sar_{duration_seconds}s_{file_number:04d}.sigmf-meta'

    # sample SN: 0000000000000000c66c63dc2d898983
    opt_str = f"-f {ctr_freq_hz} -l {if_lna_gain_db} -g {baseband_gain_db} -b {baseband_filter_bw_hz} -s {sample_rate_hz} -n {n_samples}  -r {data_out_path}"
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


    # rc = proc.returncode
    # print(f"hackrf_transfer finished with rc: {rc}")


    # TODO look at using the SigMFFile object, directly, instead
    meta_info_dict = {
    "global": {
        SigMFFile.DATATYPE_KEY: 'ci8',
        SigMFFile.SAMPLE_RATE_KEY: int(f'{sample_rate_hz}'),
        SigMFFile.HW_KEY: "HackRF, 6GHz 20dB LNA (usb-c), 5-element UHF yagi",
        SigMFFile.AUTHOR_KEY: 'Todd Stellanova',
        SigMFFile.VERSION_KEY: f'{sigmf.__version__}', 
        SigMFFile.DESCRIPTION_KEY: f'Beale SAR recorded using hackrf_transfer',
        SigMFFile.RECORDER_KEY: 'hackrf_transfer',
        'antenna:type': 'yagi',
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
            SigMFFile.LABEL_KEY: f'Beale SAR',
        }
    ]
    }

    meta_json = json.dumps(meta_info_dict, indent=2)
    # print(meta_json)
    
    with open(meta_out_path, "w") as meta_outfile:
        meta_outfile.write(meta_json)


if __name__ == "__main__":
    main()
