#pragma once
#include <omnetpp.h>

using namespace omnetpp;

/** Paced packet source. The sending rate is set from outside (by AgentBridge). */
class RateSender : public cSimpleModule
{
  private:
    cMessage *sendTimer = nullptr;
    double packetBits = 12000;       // 1500-byte data packets
    double rateBps = 0;
    simtime_t lastSendTime = -1;
    bool running = false;
    // per-monitoring-interval counters
    double sentBits = 0, rttSum = 0, rttLast = -1;
    long ackCount = 0;
    double lastKnownRtt = -1;        // survives interval resets (used when no ACK arrives in an interval)

    void scheduleNext();

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
    virtual void refreshDisplay() const override;

  public:
    virtual ~RateSender();
    void configure(double packetBits_) { packetBits = packetBits_; }
    void start();
    void setRateMbps(double mbps);
    double getRateMbps() const { return rateBps / 1e6; }
    double getSentBits() const { return sentBits; }
    long getAckCount() const { return ackCount; }
    double getRttLast() const { return rttLast >= 0 ? rttLast : lastKnownRtt; }
    double getRttMean() const { return ackCount > 0 ? rttSum / ackCount : getRttLast(); }
    void resetIntervalCounters() { sentBits = 0; rttSum = 0; ackCount = 0; rttLast = -1; }
};
