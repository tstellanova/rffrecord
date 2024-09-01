#!/usr/bin/env python
"""
Record RF baseband from a particular band/code
using an IP-connected Pluto SDR, and output in SigMF format
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

def freq_ctr_and_bw(bandcode):
    """
    Convert a band code to the corresponding center frequency and bandwidths (all in float MHz),
    for example:
    GPS L1 Band: 1575.42 MHz with a bandwidth of 15.345 MHz
    GPS L2 Band: 1227.6 MHz with a bandwidth of 11 MHz
    GPS L5 Band: 1176.45 MHz with a bandwidth of 12.5 MHz
    Note that these bandwidths are the full allocated bandwidth: receiving a narrower bandwidth may be sufficient.
    Refer to ESA GPS band plan notes: https://gssc.esa.int/navipedia/index.php/GPS_Signal_Plan

    :param bandcode: A short string code for the band, eg 'L5' or 'L1CA'
    :return: (center_freq_mhz, full_bandwidth_mhz, sampling_bandwidth_mhz, baseband_filter_mhz)
    """
    match bandcode:
        case 'L2':
            return 1227.6000, 11.000, 3.0, 3.0 # adequate for most L2 GPS uses?
        case 'L2C':
            # The exact L2C chip rate is 1.023 Msps (2x 511.5 Ksps multiplexed)
            # "The CM code is 10,230 chips long, repeating every 20 ms.
            #  The CL code is 767,250 chips long, repeating every 1,500 ms.
            #  Each signal is transmitted at 511,500 chips per second (chip/s);
            # however, they are multiplexed together to form a 1,023,000-chip/s signal."
            return 1227.6000, 11.000, 5.0, 5.0
        case 'L3':
            return 1381.0500, 15.345, 2.0, 2.0
        case 'L4':
            return 1379.9133, 15.345, 2.0, 2.0
        case 'L5':
            # L5 has 10.23 MHz chipping rate (10.23E6 chips /sec)
            # L5 has a code length of 10.23E3 chips
            return 1176.4500, 12.500, 12.50, 12.5
        case 'L5I':
            return 1176.4500, 20.4600, 20.0, 20.0
        case 'B2a':
            return 1176.4500, 12.500, 2.0, 2.0
        case 'E5a':
            return 1176.4500, 20.460, 2.0, 2.0
        case 'E5b' | 'B2b':
            return 1207.1400, 20.460, 2.0, 2.0
        case 'E6':
            return 1278.7500, 40.920, 2.0, 2.0
        case 'B3':
            return 1286.5300, 10.000, 2.0, 2.0
        case 'E1':
            return 1575.4200, 24.552, 2.0, 2.0
        case 'H1':
            # Record the Hydrogen Line
            return 1420.4000, 12.000, 2.0, 2.0
        case 'RCM':
            # Record on the Radarsat Constellation (RCM-1, RCM-2, RCM-3, Radarsat-2)
            return 5405.000, 100.00, 25.0, 25.0
        case 'L1':
            # adequate for most L1 GPS uses
            return 1575.4200, 15.345, 2.0, 2.0
        case 'L1CA':
            # exact chip rate for C/A is 1.023E6 chips/sec
            # number of chips is 1023
            # seconds/chip is 1E-6, sampled at Nyquist: 2 MHz
            return 1575.4200, 15.345, 4.0, 2.0
        case 'L1PY':
            # chip rate for P(Y) code is 10.23E6 chips/sec?
            # number of chips is 10230 ?
            return 1575.4200, 15.345, 10.23, 10.0
        case 'L1M':
            # Exact chip rate for M code is 5.115 MHz, and sub-carrier is 10.23 MHz?
            return 1575.4200, 15.345, 10.23, 5.115
        case 'V1':
            # simulate receving a "virtual" white noise signal on the L1 band
            return 1575.4200, 24.0, 10.0, 10.0
        case 'KALX' | _:
            return 90.7000, 5.000, 2.0, 0.2 # KALX.berkeley.edu


def main():
    parser = argparse.ArgumentParser(description='Grab some baseband RF data using a remote Pluto SDR over IP')
    parser.add_argument('--band', '-b', dest='bandcode', default='L1',
                        choices=['L1', 'L2', 'L3', 'L4', 'L5',
                                 'L1CA','L1M', 'L2C', 'L2CM', 'L5I',
                                 'B2a', 'B2c', 'B3',
                                 'E1', 'E5a', 'E5b', 'E6',
                                 'H1','KALX', 'V1', 'RCM'
                                 ],
                        help='Short code string for the band selected (default: L1)')
    parser.add_argument('--ip_address','-ip', dest='ip_address',required=True,
                        help='IP address of the Pluto to connect with')
    parser.add_argument('--satdump_bin_path',dest='satdump_bin_path',
                        default='/Applications/SatDump.app/Contents/MacOS/satdump',
                        help="System path where the satdump binary is found")
    parser.add_argument("--out_path",dest='out_path',default='./',
                        help="Directory path to place output files"
                        )
    parser.add_argument('--duration', '-d',  type=int, default=10,
                        help='Duration to capture, in seconds')


    args = parser.parse_args()
    bandcode = args.bandcode
    duration_seconds = args.duration
    sdr_ip_address = args.ip_address
    satdump_bin_path = args.satdump_bin_path
    out_path = args.out_path


    if not os.path.isdir(out_path):
        print(f"out_path {out_path} does not exist")
        return -1

    print(f'band {bandcode} duration {duration_seconds}')

    freq_ctr_mhz, full_bandwidth, sampling_bw_mhz, _ = freq_ctr_and_bw(bandcode)
    print(f"ctr: {freq_ctr_mhz} bw: {sampling_bw_mhz}")

    if sampling_bw_mhz < 2.0:
        sampling_bw_mhz = 2.0
    sample_rate_hz = int(sampling_bw_mhz * 1E6)
    ctr_freq_hz = int(freq_ctr_mhz * 1E6)

    n_samples = int(duration_seconds * sample_rate_hz)

    half_bw_hz = int(sample_rate_hz / 2)
    freq_lower_edge = int(ctr_freq_hz - half_bw_hz)
    freq_upper_edge = int(ctr_freq_hz + half_bw_hz)

    # figure out where to put the output files automatically
    # SatDump's native output file format would be something like:
    # 2024-08-09_03-55-00_10000000SPS_1176000000Hz.cs16

    # date_time_str = datetime.utcnow().isoformat()+'Z'
    date_time_str = datetime.utcnow().isoformat(sep='_',timespec='seconds')+'Z'

    print(f"datttsr {date_time_str}")
    full_file_path = f'{out_path}{date_time_str}'
    base_data_out_path = f'{full_file_path}_plutosat_{bandcode}_{duration_seconds}s'
    meta_out_path = f'{base_data_out_path}.sigmf-meta'
    interim_data_out_path = f'{base_data_out_path}.cs16' # satdump likes to append its own file extension
    full_data_out_path = f'{base_data_out_path}.sigmf-data'

    # gain_mode 3 == hybrid
    record_duration = duration_seconds + 1
    opt_str = (f"record {base_data_out_path} --source plutosdr --frequency {ctr_freq_hz} "
                f"--samplerate {sample_rate_hz} --baseband_format cs16  --gain_mode 3 --gain 40 "
                f"--ip_address {sdr_ip_address} --auto_reconnect true --timeout {record_duration}")

    cmd_str = f"{satdump_bin_path} {opt_str}"
    print(f"START:\n{cmd_str} ")

    # Regex to match and extract numeric values
    regex = r"[-+]?\d*\.\d+|\d+"

    total_power = float(0)
    step_count = 0
    line_count = 0
    capture_start_utc = None
    if bandcode != 'V1':
        with (Popen([cmd_str], stdout=PIPE, stderr=STDOUT, text=True, shell=True) as proc):
            for line in proc.stdout:
                if capture_start_utc is None:
                    capture_start_utc = datetime.utcnow().replace(microsecond=0).isoformat('_', )+'Z'
                line_count += 1
    else:
        # save a fake signal file
        total_interleaved_values = n_samples * 2
        # I and Q values should be completely independent
        one_big_chunk = np.random.random(total_interleaved_values)*np.iinfo(np.int8).max
        one_big_chunk.astype('int16').tofile(full_data_out_path)
        total_power = -1.0
        step_count = 1

    rc = proc.returncode
    print(f"recorder finished with rc: {rc}")
    if os.path.isfile(interim_data_out_path):
        os.rename(interim_data_out_path, full_data_out_path)


    # TODO look at using the SigMFFile object, directly, instead
    meta_info_dict = {
        "global": {
            SigMFFile.DATATYPE_KEY: 'ci16_le',
            SigMFFile.SAMPLE_RATE_KEY: int(f'{sample_rate_hz}'),
            SigMFFile.HW_KEY: "PlutoPlus SDR, bias tee, LNA",
            SigMFFile.AUTHOR_KEY: 'Todd Stellanova',
            SigMFFile.VERSION_KEY: f'{sigmf.__version__}',
            SigMFFile.DESCRIPTION_KEY: f'Band {bandcode} recorded using satdump',
            SigMFFile.RECORDER_KEY: 'satdump',
            'antenna:type': 'Wideband',
            'stellanovat:sdr': 'PlutoPlus',
            'stellanovat:LNA': 'Unknown',
            'stellanovat:LNA_pwr': 'bias_tee',
            'stellanovat:boost_amp': 'None',
        },
        "captures": [
            {
                SigMFFile.START_INDEX_KEY: 0,
                SigMFFile.FREQUENCY_KEY: int(f'{ctr_freq_hz}'),
                SigMFFile.DATETIME_KEY: f'{capture_start_utc}',
                'stellanovat:sdr_amp_mode': 'hybrid',
                'stellanovat:sdr_rx_amp_enabled': 1,
                "stellanovat:recorder_command": f'{cmd_str}',
            }
        ],
        "annotations": [
            {
                SigMFFile.START_INDEX_KEY: 0,
                SigMFFile.LENGTH_INDEX_KEY: int(f'{n_samples}'),
                SigMFFile.FHI_KEY: int(f'{freq_upper_edge}'),
                SigMFFile.FLO_KEY: int(f'{freq_lower_edge}'),
                SigMFFile.LABEL_KEY: f'Band {bandcode}',
            }
        ]
    }

    meta_json = json.dumps(meta_info_dict, indent=2)
    # print(meta_json)

    with open(meta_out_path, "w") as meta_outfile:
        meta_outfile.write(meta_json)

    print(f"wrote {meta_out_path}")

if __name__ == "__main__":
    main()
