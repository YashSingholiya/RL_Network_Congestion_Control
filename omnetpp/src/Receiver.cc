#include "Receiver.h"
#include <cstdio>

Define_Module(Receiver);

void Receiver::handleMessage(cMessage *msg)
{
    cPacket *pkt = check_and_cast<cPacket *>(msg);
    deliveredBits += pkt->getBitLength();
    cMessage *ack = new cMessage("ack");
    ack->setTimestamp(pkt->getTimestamp());          // original send time -> the sender computes the RTT
    sendDelayed(ack, retDelay, "ackOut");
    delete pkt;
}

void Receiver::refreshDisplay() const
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "delivered %.2f Mbit", deliveredBits / 1e6);
    getDisplayString().setTagArg("t", 0, buf);
}
