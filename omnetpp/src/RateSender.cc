#include "RateSender.h"
#include <cstdio>

Define_Module(RateSender);

RateSender::~RateSender()
{
    cancelAndDelete(sendTimer);
}

void RateSender::initialize()
{
    sendTimer = new cMessage("sendTimer");
    WATCH(rateBps);
    WATCH(sentBits);
    WATCH(lastKnownRtt);
}

void RateSender::start()
{
    Enter_Method_Silent();                         // called by AgentBridge: switch the context to this module
    running = true;
    if (rateBps > 0 && !sendTimer->isScheduled())
        scheduleNext();
}

void RateSender::setRateMbps(double mbps)
{
    Enter_Method_Silent();                         // called by AgentBridge: switch the context to this module
    rateBps = mbps * 1e6;
    if (!running || rateBps <= 0)
        return;
    // re-pace: the next packet is one (new) gap after the previous one, but never in the past
    if (sendTimer->isScheduled())
        cancelEvent(sendTimer);
    scheduleNext();
}

void RateSender::scheduleNext()
{
    simtime_t t = simTime();
    if (lastSendTime >= 0) {
        simtime_t earliest = lastSendTime + SimTime(packetBits / rateBps);
        if (earliest > t)
            t = earliest;
    }
    scheduleAt(t, sendTimer);
}

void RateSender::handleMessage(cMessage *msg)
{
    if (msg == sendTimer) {
        cPacket *pkt = new cPacket("data");
        pkt->setBitLength((int64_t)packetBits);
        pkt->setTimestamp(simTime());              // used by the receiver's ACK to measure the RTT
        send(pkt, "out");
        sentBits += packetBits;
        lastSendTime = simTime();
        if (running && rateBps > 0)
            scheduleNext();
    }
    else {                                          // an ACK came back
        double rtt = (simTime() - msg->getTimestamp()).dbl();
        rttSum += rtt;
        rttLast = rtt;
        lastKnownRtt = rtt;
        ackCount++;
        delete msg;
    }
}

void RateSender::finish()
{
    running = false;
}

void RateSender::refreshDisplay() const
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "rate %.2f Mbps", rateBps / 1e6);
    getDisplayString().setTagArg("t", 0, buf);
}
