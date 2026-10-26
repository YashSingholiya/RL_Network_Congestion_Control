#pragma once
#include <omnetpp.h>
#include <string>
#include "RateSender.h"
#include "BottleneckQueue.h"
#include "Receiver.h"

using namespace omnetpp;

/**
 * Lock-step bridge to the Python environment (text lines over TCP, OMNeT++ is the client).
 *
 *   OMNeT++ -> Python : HELLO
 *   Python  -> OMNeT++: CONFIG base_rtt buffer_bits packet_bits dt steps n_cap cap_0 ... cap_{n-1}
 *   OMNeT++ -> Python : READY
 *   Python  -> OMNeT++: RATE mbps                     (rate for the first monitoring interval)
 *   ... every dt of simulated time:
 *   OMNeT++ -> Python : OBS step sent_bits served_bits dropped_bits queue_bits rtt_last rtt_mean acks cap_mbps done
 *   Python  -> OMNeT++: RATE mbps                     (not sent after the last interval: done = 1)
 *
 * Simulated time does not advance while OMNeT++ waits for Python.
 */
class AgentBridge : public cSimpleModule
{
  private:
    int sock = -1;
    std::string rbuf;
    cMessage *miTimer = nullptr;
    int step = 0, totalSteps = 400;
    simsignal_t sigRate, sigThr, sigQueue, sigLat, sigLoss, sigCap;
    double dt = 0.05;
    RateSender *sender = nullptr;
    BottleneckQueue *queue = nullptr;
    Receiver *receiver = nullptr;

    void connectToPython();
    void sendLine(const std::string &line);
    std::string readLine();
    void closeSocket();
    double readRate();
    void emitObservation(bool done);

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
    virtual void refreshDisplay() const override;

  public:
    virtual ~AgentBridge();
};
