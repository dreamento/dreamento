import socket
import sys

class ZmaxSocket:
    
    """
    Defining the ZmaxSocket, the relavant port and IP for connecting the ZmaxCoDo
    to the server
    """
    def __init__(self, sock=None):
        self.MSGLEN = 4500 # InputBufferSize?
        self.serverConnected = False

        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        else:
            self.sock = sock

    def connect(self, host='127.0.0.1', port=8000):
        
        """
        Connect the socket.
        
        :param self: access the attributes and methods of the class
        :param host: host address.
        :param port: port number
        """
        try:
            self.sock.connect((host, port))
            self.serverConnected = True

        except socket.error as msg:
            print(f"Couldn't connect with the socket-server: {msg}.\n")
            self.serverConnected = False

    def send(self, msg):
        
        """
        Send message to the server.
        
        :param self: access the attributes and methods of the class
        :param msg: the message to be sent to the server.
        
        
        """
        totalsent = 0
        while totalsent < self.MSGLEN:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")

            totalsent = totalsent + sent
        
    def receive_completeBuffer(self):
        
        """
        Receiving the complete data buffer from the server.
        
        :param self: access the attributes and methods of the class
        
        :returns: message to carry
        """
        chunks = []
        bytes_recd = 0
        while bytes_recd < self.MSGLEN:
            # chunk = self.sock.recv(min(self.MSGLEN - bytes_recd, 5000))
            chunk = self.sock.recv(self.MSGLEN)
            if chunk == b'':
                raise RuntimeError("socket connection broken")

            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)

        msg = b''.join(chunks)
        if type == 0: # binary
            return msg

        else: # string
            return msg.decode("utf-8")
    
    def receive_oneLineBuffer(self, type=1):
        
        """
        Receiving the online buffer from the server.
        
        :param self: access the attributes and methods of the class
        :param type: data type (0: binary, 1: string)
            
        :returns: message
        """
        chunks = []
        bytes_recd = 0
        while bytes_recd < self.MSGLEN:
            # chunk = self.sock.recv(min(self.MSGLEN - bytes_recd, 5000))
            chunk = self.sock.recv(1)
            if chunk == b'':
                raise RuntimeError("socket connection broken")

            if chunk == b'\r':
                continue

            elif chunk == b'\n':
                break

            else:
                chunks.append(chunk)
                bytes_recd = bytes_recd + len(chunk)

        msg = b''.join(chunks)
        if type == 0: # binary
            return msg

        else: # string
            return msg.decode("utf-8")
    
    def sendString(self, msg):
        
        """
        Send string to the server
        
        :param self: access the attributes and methods of the class
        :param msg: message to be sent

        """
        # msg = ("%s\n" % "HELLO").encode('utf-8')
        msg = msg.encode('utf-8')
        self.sock.send(msg)

    def live_recieve(self):
        
        """
        receive data chunk online.
        
        :param self: access the attributes and methods of the class
        :returns: chunks
        """
        chunks = []
        while True:
            chunk = self.sock.recv(2048) # 1024?
            if chunk == b'':
                raise RuntimeError("socket connection broken")

            print(chunks)

        return b''.join(chunks)

if __name__ == "__main__":
    s = ZmaxSocket()
    s.connect('127.0.0.1',8000)
    s.sendString('HELLO\n')
    
    for i in range(500):
        rec = s.receive_oneLineBuffer()
        print(rec)
    
    