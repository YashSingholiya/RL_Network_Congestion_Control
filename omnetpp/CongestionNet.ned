//
// Packet-level model of the project's network: ONE sender -> ONE finite drop-tail buffer served by a
// bottleneck whose capacity changes over time -> ONE receiver (ACKs return to the sender).
// The RL agent (Python) sets the sending rate every monitoring interval through the AgentBridge module.
//
simple RateSender
{
    parameters:
        @display("i=block/source;p=100,150;t=,t");
    gates:
        input ackIn;
        output out;
}

simple BottleneckQueue
{
    parameters:
        @display("i=block/queue;p=300,150;t=,t");
    gates:
        input in;
        output out;
}

simple Receiver
{
    parameters:
        @display("i=block/sink;p=500,150;t=,t");
    gates:
        input in;
        output ackOut;
}

simple AgentBridge
{
    parameters:
        string host = default("127.0.0.1");   // where the Python environment listens
        int port;                              // TCP port chosen by Python for this episode
        @signal[rateMbps](type=double);
        @signal[throughputMbps](type=double);
        @signal[queueMbit](type=double);
        @signal[latencyMs](type=double);
        @signal[lossPercent](type=double);
        @signal[capacityMbps](type=double);
        @statistic[rateMbps](title="sending rate (Mbps)"; record=vector; interpolationmode=sample-hold);
        @statistic[throughputMbps](title="throughput (Mbps)"; record=vector; interpolationmode=sample-hold);
        @statistic[queueMbit](title="queue (Mbit)"; record=vector; interpolationmode=sample-hold);
        @statistic[latencyMs](title="latency (ms)"; record=vector; interpolationmode=sample-hold);
        @statistic[lossPercent](title="loss (%)"; record=vector; interpolationmode=sample-hold);
        @statistic[capacityMbps](title="link capacity (Mbps)"; record=vector; interpolationmode=sample-hold);
        @display("i=block/cogwheel;p=300,40;t=,t");
}

network CongestionNet
{
    submodules:
        sender: RateSender;
        queue: BottleneckQueue;
        receiver: Receiver;
        bridge: AgentBridge;
    connections:
        sender.out --> queue.in;
        queue.out --> receiver.in;
        receiver.ackOut --> sender.ackIn;
}
