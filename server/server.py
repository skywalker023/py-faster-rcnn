#!/usr/bin/env python

import datetime
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import urllib
from PIL import Image
import os
import sys
import base64

class WSHandler(tornado.websocket.WebSocketHandler):
    clients = []

    def check_origin(self, origin):
        print origin
        return True

    def open(self):
        print 'new connection'
        WSHandler.clients.append(self)

    def on_message(self, message):
        print 'message received %s' % message
        save_image(message)
        # Run our model.
        os.system('/home/ciplab/HEPC/py-faster-rcnn/tools/HEPC_app.py')
        # Read the output.
        with open('/home/ciplab/PycharmProjects/server/images/json.txt',"rb") as output:
            json_output=output.read()
        # Send to web.
        self.write_message(json_output)

    def on_close(self):
        print 'connection closed'
        WSHandler.clients.remove(self)

    @classmethod
    def write_to_clients(cls):
        print "Writing to clients"
        for client in cls.clients:
            client.write_message("Hi there!")

# Get the image from url and save the image to the demo directory.
def save_image(url):
    urllib.urlretrieve(url, "/home/ciplab/HEPC/py-faster-rcnn/data/demo/input.jpg")
    img = Image.open("/home/ciplab/HEPC/py-faster-rcnn/data/demo/input.jpg")
    img.save("/home/ciplab/HEPC/py-faster-rcnn/data/demo/input.jpg", "JPEG")

application = tornado.web.Application([
  (r'/ws', WSHandler),
])

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8081)
    #tornado.ioloop.IOLoop.instance().add_timeout(datetime.timedelta(seconds=15), WSHandler.write_to_clients)
    tornado.ioloop.IOLoop.instance().start()