#pragma once
#include <omnetpp.h>

using namespace omnetpp;

/** Sink that returns one small ACK per data packet (the ACK path adds half of the base RTT). */
class Receiver : public cSimpleModule
{
  private:
    double retDelay = 0;
    double deliveredBits = 0;

  protected:
    virtual void initialize() override { WATCH(deliveredBits); }
    virtual void handleMessage(cMessage *msg) override;
    virtual void refreshDisplay() const override;

  public:
    void configure(double retDelay_) { retDelay = retDelay_; }
    double getDeliveredBits() const { return deliveredBits; }
};
