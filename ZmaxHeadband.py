import numpy as np
import time
from ZmaxSocket import ZmaxSocket
import enum


class ZmaxDataID(enum.Enum):
    """
    Enumerating each data signal with a specific number.
    """
    eegr = 0
    
    """
    The assigned number to EEG R channel 
    """
    eegl = 1
    
    """
    The assigned number to EEG L channel 
    """
    
    dx = 2
    
    """
    The assigned number to acceleration in x direction of the IMU.
    """
    
    dy = 3
    
    """
    The assigned number to acceleration in y direction of the IMU.
    """
    
    dz = 4
    
    """
    The assigned number to acceleration in z direction of the IMU.
    """
    
    bodytemp = 5
    
    """
    The assigned number to body temperature data.
    """
    
    bat = 6
    
    """
    The assigned number to battery levels.
    """
    
    noise = 7
    
    """
    The assigned number to noise/microphone data.
    """
    
    light = 8
    
    """
    The assigned number to ambient light data.
    """
    
    nasal_l = 9
    
    """
    The assigned number to the left nasal channel.
    """
    
    nasal_r = 10
    
    """
    The assigned number to the right nasal channel.
    """
    oxy_ir_ac = 11
    oxy_r_ac = 12
    oxy_dark_ac = 13
    oxy_ir_dc = 14
    oxy_r_dc = 15
    oxy_dark_dc = 16


def connect():
    """
    Initiate the connection by calling a ZmaxSocket object.
    
    :param None: no input
    
    :returns: socket
    """
    socket = ZmaxSocket()
    socket.connect()
    print(socket.serverConnected)
    if socket.serverConnected:
        socket.sendString('HELLO\n')
        time.sleep(0.3)  # sec
        return socket
    else:
        return None


class ZmaxHeadband():
    def __init__(self):
        self.buf_size = 3 * 256  # 3 seconds at 256 frames per second (plotting can be SLOW)
        self.buf_eeg1 = np.zeros((self.buf_size, 1))
        self.buf_eeg2 = np.zeros((self.buf_size, 1))
        self.buf_dx = np.zeros((self.buf_size, 1))
        self.buf_dy = np.zeros((self.buf_size, 1))
        self.buf_dz = np.zeros((self.buf_size, 1))
        self.socket = connect()
        self.msgn = 1  # message number for sending stimulation

    def read(self, reqIDs=[0, 1]):
        """
        Reading the signal from ZmaxDataID class.
        
        output refers to a list of the desired outputs of the function for example [0,1,3] returns [eegl, eegr, dy]
        [0=eegr, 1=eegl, 2=dx, 3=dy, 4=dz, 5=bodytemp, 6=bat, 7=noise, 8=light, 9=nasal_l, 10=nasal_r, 11=oxy_ir_ac,
            12=oxy_r_ac, 13=oxy_dark_ac, 14=oxy_ir_dc, 15=oxy_r_dc, 16=oxy_dark_dc]
        
        :param self: access the attributes and methods of the class
        :param reqIDs: The ID of the data to record.
        
        :returns: reqVals
        """
        reqVals = []
        buf = self.socket.receive_oneLineBuffer()
        if str.startswith(buf, 'DEBUG'):  # ignore debugging messages from server
            pass

        else:
            if str.startswith(buf, 'D'):  # only process data packets
                p = buf.split('.')
                if (len(p) == 2):
                    buf = p[1]
                    packet_type = self.getbyteat(buf, 0)
                    if ((packet_type >= 1) and (packet_type <= 11)):  # packet type within correct range
                        if (len(buf) == 119):
                            # EEG channels
                            eegr = self.getwordat(buf, 1)
                            eegl = self.getwordat(buf, 3)
                            # Accelerometer channels
                            dx = self.getwordat(buf, 5)
                            dy = self.getwordat(buf, 7)
                            dz = self.getwordat(buf, 9)
                            # PPG channels (not plotted)
                            oxy_ir_ac = self.getwordat(buf, 27)  # requires external nasal sensor
                            oxy_r_ac = self.getwordat(buf, 25)  # requires external nasal sensor
                            oxy_dark_ac = self.getwordat(buf, 34)  # requires external nasal sensor
                            oxy_ir_dc = self.getwordat(buf, 17)  # requires external nasal sensor
                            oxy_r_dc = self.getwordat(buf, 15)  # requires external nasal sensor
                            oxy_dark_dc = self.getwordat(buf, 32)  # requires external nasal sensor
                            # other channels (not plotted)
                            bodytemp = self.getwordat(buf, 36)
                            nasal_l = self.getwordat(buf, 11)  # requires external nasal sensor
                            nasal_r = self.getwordat(buf, 13)  # requires external nasal sensor
                            light = self.getwordat(buf, 21)
                            bat = self.getwordat(buf, 23)
                            noise = self.getwordat(buf, 19)
                            # convert
                            eegr, eegl = self.ScaleEEG(eegr), self.ScaleEEG(eegl)
                            dx, dy, dz = self.ScaleAccel(dx), self.ScaleAccel(dy), self.ScaleAccel(dz)
                            bodytemp = self.BodyTemp(bodytemp)
                            bat = self.BatteryVoltage(bat)
                            # for function return
                            result = [eegr, eegl, dx, dy, dz, bodytemp, bat, noise, light, nasal_l, nasal_r, \
                                      oxy_ir_ac, oxy_r_ac, oxy_dark_ac, oxy_ir_dc, oxy_r_dc, oxy_dark_dc]
                            for i in reqIDs:
                                reqVals.append(result[i])

        return reqVals

    def getbyteat(self, buf, idx=0):
        """
        Receiving the bytes of data.
        
        for example getbyteat("08-80-56-7F-EA",0) -> hex2dec(08)
                    getbyteat("08-80-56-7F-EA",2) -> hex2dec(56)
                    
        :param self: access the attributes and methods of the class
        :param buf: the data buffer.
        :param idx: the index of the data (indicating the data type).
        
        :returns: self.hex2dec(s)

        """
        s = buf[idx * 3:idx * 3 + 2]
        return self.hex2dec(s)

    def getwordat(self, buf, idx=0):
        
        """
        Get the word
        
        :param self: access the attributes and methods of the class
        :param buf: the data buffer.
        :param idx: the index of the data (indicating the data type).
        
        :returns: w
        """
        w = self.getbyteat(buf, idx) * 256 + self.getbyteat(buf, idx + 1)
        return w

    def ScaleEEG(self, e):  # word value to uV
    
        """
        Scale the EEG data by receiving the word and converting it to uV.
        
        :param self: access the attributes and methods of the class
        :param e: word to convert
        
        :returns: d
        """
        uvRange = 3952;
        d = e - 32768;
        d = d * uvRange;
        d = d / 65536
        return d

    def ScaleAccel(self, dx):  # word value to 'g'
    
        """
        Scale the acceleration value with respecto to the gravity (g).
        
        :param self: access the attributes and methods of the class
        :param dx:  word value to be converted to 'g' scale
        """
        d = dx * 4 / 4096 - 2
        return d

    def BatteryVoltage(self, vbat):  # word value to Volts
    
        """
        The battery voltage data
        
        :param self: access the attributes and methods of the class
        :param vbat: The word to be converted to battery voltage (v)
        
        :returns: t (temperature)
        """
        
        v = vbat / 1024 * 6.60;
        return v

    def BodyTemp(self, bodytemp):  # word value to degrees C
    
        """
        The body temperature data.
        
        :param self: access the attributes and methods of the class
        :param bodytemp: The temperature in words (to be converted to degrees)
        
        :returns: t (temperature)
        """
        v = bodytemp / 1024 * 3.3
        t = 15 + ((v - 1.0446) / 0.0565537333333333)
        return t

    def hex2dec(self, s):
        """
        Return the integer value of a hexadecimal string 's'.
        
        :param self: access the attributes and methods of the class
        :param s: string value
        
        :returns: int(s, 16)
        
        """
        return int(s, 16)

    def dec2hex(self, n, pad=0):
        """
        return the hexadecimal string representation of integer 'n'
    
        :param self: access the attributes and methods of the class
        :param n: string value
        :param pad: padding 
        
        :returns: s.rjust(pad, '0')
        
        """
        s = "%X" % n
        if pad == 0:
            return s
        else:
            # for example if pad = 3, the dec2hex(5,2) = '005'
            return s.rjust(pad, '0')

    def stimulate(self, rgb1=(0, 0, 2), rgb2=(0, 0, 2), pwm1=254, pwm2=0, t1=1, t2=3, reps=5, vib=1, alt=0):
        """
        
        Stimulation trigger.
        
        example:
        LIVEMODE_SENDBYTES 15 6 111 04-00-00-02-00-00-02-FE-00-01-03-05-01-00\r\n
        command = "LIVEMODE_SENDBYTES"
        retries = 15
        msgn = 6
        retry_ms = 111
        LIVECMD_FLASHLEDS = 04
        r = 00
        g = 00
        b = 02
        r2 = 00
        g2 = 00
        b2 = 02
        pwm1 = FE (254); intensity from 2(1%) to 254(100%)
        pwm2 = 00
        t1 = 01
        t2 = 03
        reps = 05
        vib = 01
        alt = 00 # althernate eyes
        
        :param self: access the attributes and methods of the class
        :param rgb1: rgb value 1
        :param rgb2: rgb value 2
        :param pwm1: ON intensity
        :param pwm2: OFF intensity   
        :param t1: time ON
        :param t2: time OFF
        :param reps: number of blinks/ repititions
        :param vib: Vibration triggering
        :param alt: Alternating the LEDs
        """
        command = "LIVEMODE_SENDBYTES"
        retries = 15
        retry_ms = 111
        LIVECMD_FLASHLEDS = 4

        i1 = self.dec2hex(LIVECMD_FLASHLEDS, pad=2)
        i2 = self.dec2hex(rgb1[0], pad=2)
        i3 = self.dec2hex(rgb1[1], pad=2)
        i4 = self.dec2hex(rgb1[2], pad=2)
        i5 = self.dec2hex(rgb2[0], pad=2)
        i6 = self.dec2hex(rgb2[1], pad=2)
        i7 = self.dec2hex(rgb2[2], pad=2)
        i8 = self.dec2hex(pwm1, pad=2)
        i9 = self.dec2hex(pwm2, pad=2)
        i10 = self.dec2hex(t1, pad=2)
        i11 = self.dec2hex(t2, pad=2)
        i12 = self.dec2hex(reps, pad=2)
        i13 = self.dec2hex(vib, pad=2)
        i14 = self.dec2hex(alt, pad=2)

        s = f"""{command} {retries} {self.msgn} {retry_ms} {i1}-{i2}-{i3}-{i4}-{i5}-{i6}-{i7}-{i8}-{i9}-{i10}-{i11}-{i12}-{i13}-{i14}\r\n"""
        # print(s)
        self.socket.sendString(s)
        self.msgn += 1