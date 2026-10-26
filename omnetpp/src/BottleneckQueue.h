#pragma once
#include <omnetpp.h>
#include <deque>
#include <vector>

using namespace omnetpp;

/**
 * Finite drop-tail FIFO buffer in front of a bottleneck link whose capacity follows a trace
 * (one value per monitoring interval). Packets are served one at a time; the service time of a packet
 * is bits / capacity(at the start of its transmission).
 */
class BottleneckQueue : public cSimpleModule
{
  private:
    std::deque<cPacket *> fifo;
    cPacket *inService = nullptr;
    cMessage *endService = nullptr;
    double queueBits = 0;            // bits in the system: waiting packets + the packet in service
    double bufferBits = 1e9;
    double fwdDelay = 0;             // propagation delay towards the receiver (seconds)
    double dt = 0.05;
    std::vector<double> capMbps;
    simtime_t serviceStart = 0;
    double serviceDur = 0, credited = 0;
    // per-interval counters
    double servedBits = 0, droppedBits = 0;

    void startService();

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void refreshDisplay() const override;

  public:
    virtual ~BottleneckQueue();
    void configure(const std::vector<double> &caps, double dt_, double bufferBits_, double fwdDelay_);
    double capacityAtMbps(simtime_t t) const;
    double getQueueBits() const { return queueBits; }
    /** Bits served in the current interval, counting the already transmitted part of the packet in service. */
    double takeServedBits();
    double getDroppedBits() const { return droppedBits; }
    void resetIntervalCounters() { servedBits = 0; droppedBits = 0; }
};
