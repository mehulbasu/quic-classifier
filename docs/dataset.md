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