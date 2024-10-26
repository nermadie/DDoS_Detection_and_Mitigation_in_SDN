from mininet.topo import Topo
from mininet.net import Mininet
# from mininet.node import CPULimitedHost
from mininet.link import TCLink
# from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import OVSKernelSwitch, RemoteController
# from time import sleep

class MyTopo( Topo ):

    def build( self ):

        s1 = self.addSwitch( 's1', cls=OVSKernelSwitch, protocols='OpenFlow13' )

        h1 = self.addHost( 'h1', cpu=1.0/20,mac="00:00:00:00:00:01", ip="10.0.0.1/24" )
        h2 = self.addHost( 'h2', cpu=1.0/20, mac="00:00:00:00:00:02", ip="10.0.0.2/24" )
        h3 = self.addHost( 'h3', cpu=1.0/20, mac="00:00:00:00:00:03", ip="10.0.0.3/24" )

        s2 = self.addSwitch( 's2', cls=OVSKernelSwitch, protocols='OpenFlow13' )

        h4 = self.addHost( 'h4', cpu=1.0/20, mac="00:00:00:00:00:04", ip="10.0.0.4/24" )
        h5 = self.addHost( 'h5', cpu=1.0/20, mac="00:00:00:00:00:05", ip="10.0.0.5/24" )
        h6 = self.addHost( 'h6', cpu=1.0/20, mac="00:00:00:00:00:06", ip="10.0.0.6/24" )

        s3 = self.addSwitch( 's3', cls=OVSKernelSwitch, protocols='OpenFlow13' )

        h7 = self.addHost( 'h7', cpu=1.0/20, mac="00:00:00:00:00:07", ip="10.0.0.7/24" )
        h8 = self.addHost( 'h8', cpu=1.0/20, mac="00:00:00:00:00:08", ip="10.0.0.8/24" )
        h9 = self.addHost( 'h9', cpu=1.0/20, mac="00:00:00:00:00:09", ip="10.0.0.9/24" )

        s4 = self.addSwitch( 's4', cls=OVSKernelSwitch, protocols='OpenFlow13' )

        h10 = self.addHost( 'h10', cpu=1.0/20, mac="00:00:00:00:00:10", ip="10.0.0.10/24" )
        h11 = self.addHost( 'h11', cpu=1.0/20, mac="00:00:00:00:00:11", ip="10.0.0.11/24" )
        h12 = self.addHost( 'h12', cpu=1.0/20, mac="00:00:00:00:00:12", ip="10.0.0.12/24" )

        # Add links
        self.addLink( h1, s1, bw=10)
        self.addLink( h2, s1, bw=10)
        self.addLink( h3, s1, bw=10)

        self.addLink( h4, s2, bw=10)
        self.addLink( h5, s2, bw=10)
        self.addLink( h6, s2, bw=10)

        self.addLink( h7, s3, bw=10)
        self.addLink( h8, s3, bw=10)
        self.addLink( h9, s3, bw=10)

        self.addLink( h10, s4, bw=10)
        self.addLink( h11, s4, bw=10)
        self.addLink( h12, s4, bw=10)

        self.addLink( s1, s2 )
        self.addLink( s2, s3 )
        self.addLink( s3, s4 )

def startNetwork():

    topo = MyTopo()
    c0 = RemoteController('c1', ip='10.10.52.202', port=6653)

    controller_ip = None
    with open("../controller_ip.txt", 'r') as file:
        controller_ip = file.readline().strip()

    c0 = RemoteController('c0', ip=controller_ip, port=6653)
    net = Mininet(topo=topo, link=TCLink, controller=c0)

    net.start()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    startNetwork()
