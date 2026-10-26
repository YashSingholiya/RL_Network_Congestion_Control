#include "AgentBridge.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

Define_Module(AgentBridge);

AgentBridge::~AgentBridge()
{
    cancelAndDelete(miTimer);
    closeSocket();
}

void AgentBridge::closeSocket()
{
    if (sock >= 0) {
        ::close(sock);
        sock = -1;
    }
}

void AgentBridge::connectToPython()
{
    std::string host = par("host").stdstringValue();
    int port = par("port");
    sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
        throw cRuntimeError("AgentBridge: cannot create socket");
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
        throw cRuntimeError("AgentBridge: bad host address '%s'", host.c_str());
    for (int attempt = 0; ; attempt++) {                       // the Python server is already listening
        if (::connect(sock, (sockaddr *)&addr, sizeof(addr)) == 0)
            break;
        if (attempt >= 50)
            throw cRuntimeError("AgentBridge: cannot connect to %s:%d", host.c_str(), port);
        ::usleep(100000);
    }
    int one = 1;
    ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

void AgentBridge::sendLine(const std::string &line)
{
    std::string out = line + "\n";
    size_t sent = 0;
    while (sent < out.size()) {
        ssize_t n = ::send(sock, out.data() + sent, out.size() - sent, 0);
        if (n <= 0)
            throw cRuntimeError("AgentBridge: connection to Python lost while sending");
        sent += (size_t)n;
    }
}

std::string AgentBridge::readLine()
{
    for (;;) {
        size_t pos = rbuf.find('\n');
        if (pos != std::string::npos) {
            std::string line = rbuf.substr(0, pos);
            rbuf.erase(0, pos + 1);
            return line;
        }
        char buf[4096];
        ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
        if (n <= 0)
            throw cRuntimeError("AgentBridge: connection to Python closed");
        rbuf.append(buf, (size_t)n);
    }
}

double AgentBridge::readRate()
{
    std::istringstream is(readLine());
    std::string tag;
    double rate = -1;
    is >> tag >> rate;
    if (tag == "CLOSE")
        throw cRuntimeError("AgentBridge: Python closed the episode");
    if (tag != "RATE" || !(rate >= 0))
        throw cRuntimeError("AgentBridge: expected 'RATE <mbps>' from Python");
    return rate;
}

void AgentBridge::initialize()
{
    cModule *net = getParentModule();
    sender = check_and_cast<RateSender *>(net->getSubmodule("sender"));
    queue = check_and_cast<BottleneckQueue *>(net->getSubmodule("queue"));
    receiver = check_and_cast<Receiver *>(net->getSubmodule("receiver"));

    WATCH(step);
    sigRate = registerSignal("rateMbps");
    sigThr = registerSignal("throughputMbps");
    sigQueue = registerSignal("queueMbit");
    sigLat = registerSignal("latencyMs");
    sigLoss = registerSignal("lossPercent");
    sigCap = registerSignal("capacityMbps");

    connectToPython();
    sendLine("HELLO");

    std::istringstream is(readLine());                          // CONFIG line
    std::string tag;
    double baseRtt, bufferBits, packetBits;
    int nCap;
    is >> tag >> baseRtt >> bufferBits >> packetBits >> dt >> totalSteps >> nCap;
    if (tag != "CONFIG" || !is)
        throw cRuntimeError("AgentBridge: malformed CONFIG line");
    std::vector<double> caps(nCap);
    for (int i = 0; i < nCap; i++)
        is >> caps[i];
    if (!is)
        throw cRuntimeError("AgentBridge: CONFIG line has fewer capacity values than announced");

    sender->configure(packetBits);
    queue->configure(caps, dt, bufferBits, baseRtt / 2);        // forward half of the base RTT
    receiver->configure(baseRtt / 2);                           // ACK path: the other half

    sendLine("READY");
    sender->setRateMbps(readRate());                            // rate for the first interval
    sender->start();
    miTimer = new cMessage("monitoringInterval");
    scheduleAt(simTime() + SimTime(dt), miTimer);
}

void AgentBridge::emitObservation(bool done)
{
    double served = queue->takeServedBits();
    double capNow = queue->capacityAtMbps(simTime() - SimTime(dt) / 2);   // capacity used during this interval
    char buf[512];
    std::snprintf(buf, sizeof(buf), "OBS %d %.9g %.9g %.9g %.9g %.9g %.9g %ld %.9g %d", step,
                  sender->getSentBits(), served, queue->getDroppedBits(), queue->getQueueBits(),
                  sender->getRttLast(), sender->getRttMean(), sender->getAckCount(), capNow, done ? 1 : 0);
    sendLine(buf);
    // the same numbers as OMNeT++ result vectors (recorded when vector recording is on, see the "Record" config)
    double sentBits = sender->getSentBits(), rtt = sender->getRttLast();
    emit(sigRate, sender->getRateMbps());
    emit(sigThr, served / 1e6 / dt);
    emit(sigQueue, queue->getQueueBits() / 1e6);
    if (rtt >= 0)
        emit(sigLat, rtt * 1000);
    emit(sigLoss, sentBits > 0 ? 100.0 * queue->getDroppedBits() / sentBits : 0.0);
    emit(sigCap, capNow);
    sender->resetIntervalCounters();
    queue->resetIntervalCounters();
}

void AgentBridge::handleMessage(cMessage *msg)
{
    step++;
    bool done = step >= totalSteps;
    emitObservation(done);
    if (done) {
        closeSocket();
        endSimulation();
        return;
    }
    sender->setRateMbps(readRate());                            // blocks until Python has decided
    scheduleAt(simTime() + SimTime(dt), miTimer);
}

void AgentBridge::finish()
{
    closeSocket();
}

void AgentBridge::refreshDisplay() const
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "step %d / %d\n(Python decides the rate)", step, totalSteps);
    getDisplayString().setTagArg("t", 0, buf);
}
