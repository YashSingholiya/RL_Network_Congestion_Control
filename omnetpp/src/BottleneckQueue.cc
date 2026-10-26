#include "BottleneckQueue.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

Define_Module(BottleneckQueue);

BottleneckQueue::~BottleneckQueue()
{
    cancelAndDelete(endService);
    delete inService;
    for (auto *p : fifo)
        delete p;
}

void BottleneckQueue::initialize()
{
    endService = new cMessage("endService");
    WATCH(queueBits);
    WATCH(bufferBits);
    WATCH(servedBits);
    WATCH(droppedBits);
}

void BottleneckQueue::configure(const std::vector<double> &caps, double dt_, double bufferBits_, double fwdDelay_)
{
    capMbps = caps;
    dt = dt_;
    bufferBits = bufferBits_;
    fwdDelay = fwdDelay_;
}

double BottleneckQueue::capacityAtMbps(simtime_t t) const
{
    if (capMbps.empty())
        return 1.0;
    size_t idx = (size_t)std::floor(t.dbl() / dt + 1e-9);
    return capMbps[std::min(idx, capMbps.size() - 1)];
}

void BottleneckQueue::startService()
{
    inService = fifo.front();
    fifo.pop_front();
    serviceStart = simTime();
    serviceDur = inService->getBitLength() / (capacityAtMbps(simTime()) * 1e6);
    credited = 0;
    scheduleAt(simTime() + serviceDur, endService);
}

double BottleneckQueue::takeServedBits()
{
    // fluid-style accounting: add the part of the packet in service that was transmitted so far
    if (inService != nullptr && serviceDur > 0) {
        double bits = inService->getBitLength();
        double done = std::min(bits, bits * (simTime() - serviceStart).dbl() / serviceDur);
        servedBits += done - credited;
        credited = done;
    }
    return servedBits;
}

void BottleneckQueue::handleMessage(cMessage *msg)
{
    if (msg == endService) {
        double bits = inService->getBitLength();
        servedBits += bits - credited;
        credited = 0;
        queueBits -= bits;
        sendDelayed(inService, fwdDelay, "out");
        inService = nullptr;
        if (!fifo.empty())
            startService();
        return;
    }
    cPacket *pkt = check_and_cast<cPacket *>(msg);
    double bits = pkt->getBitLength();
    if (queueBits + bits > bufferBits) {            // drop-tail
        droppedBits += bits;
        delete pkt;
        return;
    }
    queueBits += bits;
    fifo.push_back(pkt);
    if (inService == nullptr)
        startService();
}

void BottleneckQueue::refreshDisplay() const
{
    double fill = bufferBits > 0 ? queueBits / bufferBits : 0;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "queue %.0f%% (%.2f Mbit)\nlink %.1f Mbps", 100 * fill, queueBits / 1e6, capacityAtMbps(simTime()));
    getDisplayString().setTagArg("t", 0, buf);
    getDisplayString().setTagArg("i", 1, fill > 0.9 ? "red" : (fill > 0.5 ? "gold" : "green"));   // icon colour = buffer fill
    getDisplayString().setTagArg("i", 2, 60);
}
