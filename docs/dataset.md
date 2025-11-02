## Data description
The dataset consists of network flows describing encrypted QUIC communications. Flows were created using ipfixprobe flow exporter and are extended with packet metadata sequences, packet histograms, and with fields extracted from the QUIC Initial Packet, which is the first packet of the QUIC connection handshake. The extracted handshake fields are the Server Name Indication (SNI) domain, the used version of the QUIC protocol, and the user agent string that is available in a subset of QUIC communications.

## Packet sequences
Flows in the dataset are extended with sequences of packet sizes, directions, and inter-packet times. For the packet sizes, we consider payload size after transport headers (UDP headers for the QUIC case). Packet directions are encoded as ±1: +1 meaning a packet sent from client to server, and -1 a packet from server to client. Inter-packet times depend on the location of communicating hosts, their distance, and on the network conditions on the path. However, it is still possible to extract relevant information that correlates with user interactions and, for example, with the time required for an API/server/database to process the received data and generate the response to be sent in the next packet. Packet metadata sequences have a length of 30, which is the default setting of the used flow exporter. We also derive three fields from each packet sequence: its length, time duration, and the number of roundtrips. The roundtrips are counted as the number of changes in the communication direction (from packet directions data); in other words, each client request and server response pair counts as one roundtrip.

## Flow statistics
Flows also include standard flow statistics, which represent aggregated information about the entire bidirectional flow. The fields are: the number of transmitted bytes and packets in both directions, the duration of flow, and packet histograms. Packet histograms include binned counts of packet sizes and inter-packet times of the entire flow in both directions (more information in the PHISTS plugin documentation). There are eight bins with a logarithmic scale; the intervals are 0-15, 16-31, 32-63, 64-127, 128-255, 256-511, 512-1024, >1024 [ms or B]. The units are milliseconds for inter-packet times and bytes for packet sizes. Moreover, each flow has its end reason - either it was idle, reached the active timeout, or ended due to other reasons. This corresponds with the official IANA IPFIX-specified values. The FLOW_ENDREASON_OTHER field represents the forced end and lack of resources reasons. The end of flow detected reason is not considered because it is not relevant for UDP connections.

## Dataset structure
The dataset flows are delivered in compressed CSV files. CSV files contain one flow per row; data columns are summarized in the provided list below. For each flow data file, there is a JSON file with the number of saved and seen (before sampling) flows per service and total counts of all received (observed on the CESNET2 network), service (belonging to one of the dataset's services), and saved (provided in the dataset) flows. There is also the `stats-week.json` file aggregating flow counts of a whole week and the `stats-dataset.json` file aggregating flow counts for the entire dataset. Flow counts before sampling can be used to compute sampling ratios of individual services and to resample the dataset back to the original service distribution. Moreover, various dataset statistics, such as feature distributions and value counts of QUIC versions and user agents, are provided in the `dataset-statistics` folder. The mapping between services and service providers is provided in the `servicemap.csv` file, which also includes SNI domains used for ground truth labeling. The following list describes flow data fields in CSV files:
- ID: Unique identifier
- SRC_IP: Source IP address
- DST_IP: Destination IP address
- DST_ASN: Destination Autonomous System number
- SRC_PORT: Source port
- DST_PORT: Destination port
- PROTOCOL: Transport protocol
- QUIC_VERSION QUIC: protocol version
- QUIC_SNI: Server Name Indication domain
- QUIC_USER_AGENT: User agent string, if available in the QUIC Initial Packet
- TIME_FIRST: Timestamp of the first packet in format YYYY-MM-DDTHH-MM-SS.ffffff
- TIME_LAST: Timestamp of the last packet in format YYYY-MM-DDTHH-MM-SS.ffffff
- DURATION: Duration of the flow in seconds
- BYTES: Number of transmitted bytes from client to server
- BYTES_REV: Number of transmitted bytes from server to client
- PACKETS: Number of packets transmitted from client to server
- PACKETS_REV: Number of packets transmitted from server to client
- PPI: Packet metadata sequence in the format: [[inter-packet times], [packet directions], [packet sizes]]
- PPI_LEN: Number of packets in the PPI sequence
- PPI_DURATION: Duration of the PPI sequence in seconds
- PPI_ROUNDTRIPS: Number of roundtrips in the PPI sequence
- PHIST_SRC_SIZES: Histogram of packet sizes from client to server
- PHIST_DST_SIZES: Histogram of packet sizes from server to client
- PHIST_SRC_IPT: Histogram of inter-packet times from client to server
- PHIST_DST_IPT: Histogram of inter-packet times from server to client
- APP: Web service label
- CATEGORY: Service category
- FLOW_ENDREASON_IDLE: Flow was terminated because it was idle
- FLOW_ENDREASON_ACTIVE: Flow was terminated because it reached the active timeout
- FLOW_ENDREASON_OTHER: Flow was terminated for other reasons

## Zip file structure (cesnet-quic22.zip)
```
Archive:  cesnet-quic22.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
        0  2022-12-07 07:30   cesnet-quic22/
    10530  2022-12-07 07:39   cesnet-quic22/README.md
        0  2022-12-06 13:37   cesnet-quic22/W-2022-47/
        0  2022-12-07 06:49   cesnet-quic22/W-2022-47/1_Mon/
1133587001  2022-12-06 12:42   cesnet-quic22/W-2022-47/1_Mon/flows-20221121.csv.gz
     9650  2022-12-06 12:42   cesnet-quic22/W-2022-47/1_Mon/stats-20221121.json
        0  2022-12-07 07:13   cesnet-quic22/W-2022-47/6_Sat/
335058896  2022-12-06 13:32   cesnet-quic22/W-2022-47/6_Sat/flows-20221126.csv.gz
     9530  2022-12-06 13:32   cesnet-quic22/W-2022-47/6_Sat/stats-20221126.json
        0  2022-12-07 07:02   cesnet-quic22/W-2022-47/3_Wed/
1170723649  2022-12-06 13:08   cesnet-quic22/W-2022-47/3_Wed/flows-20221123.csv.gz
     9655  2022-12-06 13:08   cesnet-quic22/W-2022-47/3_Wed/stats-20221123.json
        0  2022-12-07 07:11   cesnet-quic22/W-2022-47/5_Fri/
757980663  2022-12-06 13:28   cesnet-quic22/W-2022-47/5_Fri/flows-20221125.csv.gz
     9620  2022-12-06 13:28   cesnet-quic22/W-2022-47/5_Fri/stats-20221125.json
        0  2022-12-07 07:15   cesnet-quic22/W-2022-47/7_Sun/
412743415  2022-12-06 13:37   cesnet-quic22/W-2022-47/7_Sun/flows-20221127.csv.gz
     9550  2022-12-06 13:37   cesnet-quic22/W-2022-47/7_Sun/stats-20221127.json
        0  2022-12-07 07:07   cesnet-quic22/W-2022-47/4_Thu/
     9649  2022-12-06 13:20   cesnet-quic22/W-2022-47/4_Thu/stats-20221124.json
1052404287  2022-12-06 13:20   cesnet-quic22/W-2022-47/4_Thu/flows-20221124.csv.gz
        0  2022-12-07 06:55   cesnet-quic22/W-2022-47/2_Tue/
1180352308  2022-12-06 12:55   cesnet-quic22/W-2022-47/2_Tue/flows-20221122.csv.gz
     9650  2022-12-06 12:55   cesnet-quic22/W-2022-47/2_Tue/stats-20221122.json
     9811  2022-12-06 13:37   cesnet-quic22/W-2022-47/stats-week.json
        0  2022-12-07 05:28   cesnet-quic22/dataset-statistics/
      154  2022-12-07 05:14   cesnet-quic22/dataset-statistics/quic-version.csv
      311  2022-12-06 13:37   cesnet-quic22/dataset-statistics/week-stats.csv
     2602  2022-12-07 05:14   cesnet-quic22/dataset-statistics/app.csv
    42283  2022-12-07 05:14   cesnet-quic22/dataset-statistics/dataset-stats.pdf
      928  2022-12-07 05:14   cesnet-quic22/dataset-statistics/categories.csv
   139716  2022-12-07 05:17   cesnet-quic22/dataset-statistics/quic-ua.csv
    18438  2022-12-07 05:14   cesnet-quic22/dataset-statistics/asn.csv
        0  2022-12-06 11:39   cesnet-quic22/W-2022-45/
        0  2022-12-07 05:53   cesnet-quic22/W-2022-45/1_Mon/
     9648  2022-12-06 10:46   cesnet-quic22/W-2022-45/1_Mon/stats-20221107.json
1112037497  2022-12-06 10:46   cesnet-quic22/W-2022-45/1_Mon/flows-20221107.csv.gz
        0  2022-12-07 06:16   cesnet-quic22/W-2022-45/6_Sat/
320651572  2022-12-06 11:34   cesnet-quic22/W-2022-45/6_Sat/flows-20221112.csv.gz
     9534  2022-12-06 11:34   cesnet-quic22/W-2022-45/6_Sat/stats-20221112.json
        0  2022-12-07 06:05   cesnet-quic22/W-2022-45/3_Wed/
1109383292  2022-12-06 11:12   cesnet-quic22/W-2022-45/3_Wed/flows-20221109.csv.gz
     9648  2022-12-06 11:12   cesnet-quic22/W-2022-45/3_Wed/stats-20221109.json
        0  2022-12-07 06:14   cesnet-quic22/W-2022-45/5_Fri/
740564141  2022-12-06 11:31   cesnet-quic22/W-2022-45/5_Fri/flows-20221111.csv.gz
     9614  2022-12-06 11:31   cesnet-quic22/W-2022-45/5_Fri/stats-20221111.json
        0  2022-12-07 06:18   cesnet-quic22/W-2022-45/7_Sun/
400628790  2022-12-06 11:39   cesnet-quic22/W-2022-45/7_Sun/flows-20221113.csv.gz
     9564  2022-12-06 11:39   cesnet-quic22/W-2022-45/7_Sun/stats-20221113.json
        0  2022-12-07 06:10   cesnet-quic22/W-2022-45/4_Thu/
1017633038  2022-12-06 11:22   cesnet-quic22/W-2022-45/4_Thu/flows-20221110.csv.gz
     9648  2022-12-06 11:22   cesnet-quic22/W-2022-45/4_Thu/stats-20221110.json
        0  2022-12-07 05:59   cesnet-quic22/W-2022-45/2_Tue/
1134231227  2022-12-06 10:59   cesnet-quic22/W-2022-45/2_Tue/flows-20221108.csv.gz
     9649  2022-12-06 10:59   cesnet-quic22/W-2022-45/2_Tue/stats-20221108.json
     9811  2022-12-06 11:39   cesnet-quic22/W-2022-45/stats-week.json
        0  2022-12-06 12:30   cesnet-quic22/W-2022-46/
        0  2022-12-07 06:24   cesnet-quic22/W-2022-46/1_Mon/
1101416699  2022-12-06 11:51   cesnet-quic22/W-2022-46/1_Mon/flows-20221114.csv.gz
     9647  2022-12-06 11:51   cesnet-quic22/W-2022-46/1_Mon/stats-20221114.json
        0  2022-12-07 06:41   cesnet-quic22/W-2022-46/6_Sat/
277227093  2022-12-06 12:26   cesnet-quic22/W-2022-46/6_Sat/flows-20221119.csv.gz
     9518  2022-12-06 12:26   cesnet-quic22/W-2022-46/6_Sat/stats-20221119.json
        0  2022-12-07 06:35   cesnet-quic22/W-2022-46/3_Wed/
903548574  2022-12-06 12:13   cesnet-quic22/W-2022-46/3_Wed/flows-20221116.csv.gz
     9632  2022-12-06 12:13   cesnet-quic22/W-2022-46/3_Wed/stats-20221116.json
        0  2022-12-07 06:40   cesnet-quic22/W-2022-46/5_Fri/
     9580  2022-12-06 12:23   cesnet-quic22/W-2022-46/5_Fri/stats-20221118.json
491871841  2022-12-06 12:23   cesnet-quic22/W-2022-46/5_Fri/flows-20221118.csv.gz
        0  2022-12-07 06:43   cesnet-quic22/W-2022-46/7_Sun/
369316805  2022-12-06 12:30   cesnet-quic22/W-2022-46/7_Sun/flows-20221120.csv.gz
     9536  2022-12-06 12:30   cesnet-quic22/W-2022-46/7_Sun/stats-20221120.json
        0  2022-12-07 06:37   cesnet-quic22/W-2022-46/4_Thu/
     9544  2022-12-06 12:17   cesnet-quic22/W-2022-46/4_Thu/stats-20221117.json
366659727  2022-12-06 12:17   cesnet-quic22/W-2022-46/4_Thu/flows-20221117.csv.gz
        0  2022-12-07 06:30   cesnet-quic22/W-2022-46/2_Tue/
1101431403  2022-12-06 12:03   cesnet-quic22/W-2022-46/2_Tue/flows-20221115.csv.gz
     9642  2022-12-06 12:03   cesnet-quic22/W-2022-46/2_Tue/stats-20221115.json
     9785  2022-12-06 12:30   cesnet-quic22/W-2022-46/stats-week.json
        0  2022-12-06 10:34   cesnet-quic22/W-2022-44/
        0  2022-12-07 05:26   cesnet-quic22/W-2022-44/1_Mon/
491061964  2022-12-06 09:47   cesnet-quic22/W-2022-44/1_Mon/flows-20221031.csv.gz
     9587  2022-12-06 09:47   cesnet-quic22/W-2022-44/1_Mon/stats-20221031.json
        0  2022-12-07 05:45   cesnet-quic22/W-2022-44/6_Sat/
329532271  2022-12-06 10:30   cesnet-quic22/W-2022-44/6_Sat/flows-20221105.csv.gz
     9537  2022-12-06 10:30   cesnet-quic22/W-2022-44/6_Sat/stats-20221105.json
        0  2022-12-07 05:34   cesnet-quic22/W-2022-44/3_Wed/
     9635  2022-12-06 10:05   cesnet-quic22/W-2022-44/3_Wed/stats-20221102.json
996339615  2022-12-06 10:05   cesnet-quic22/W-2022-44/3_Wed/flows-20221102.csv.gz
        0  2022-12-07 05:43   cesnet-quic22/W-2022-44/5_Fri/
764870427  2022-12-06 10:26   cesnet-quic22/W-2022-44/5_Fri/flows-20221104.csv.gz
     9623  2022-12-06 10:26   cesnet-quic22/W-2022-44/5_Fri/stats-20221104.json
        0  2022-12-07 05:47   cesnet-quic22/W-2022-44/7_Sun/
     9549  2022-12-06 10:34   cesnet-quic22/W-2022-44/7_Sun/stats-20221106.json
400778388  2022-12-06 10:34   cesnet-quic22/W-2022-44/7_Sun/flows-20221106.csv.gz
        0  2022-12-07 05:39   cesnet-quic22/W-2022-44/4_Thu/
988521449  2022-12-06 10:17   cesnet-quic22/W-2022-44/4_Thu/flows-20221103.csv.gz
     9633  2022-12-06 10:17   cesnet-quic22/W-2022-44/4_Thu/stats-20221103.json
        0  2022-12-07 05:28   cesnet-quic22/W-2022-44/2_Tue/
     9579  2022-12-06 09:53   cesnet-quic22/W-2022-44/2_Tue/stats-20221101.json
488232319  2022-12-06 09:53   cesnet-quic22/W-2022-44/2_Tue/flows-20221101.csv.gz
     9790  2022-12-06 10:34   cesnet-quic22/W-2022-44/stats-week.json
     9925  2022-12-06 13:37   cesnet-quic22/stats-dataset.json
---------                     -------
20949321286                     103 files
```