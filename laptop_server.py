#!/usr/bin/env python3
"""Run on laptop; issue absolute joint angle targets to EV3."""
import socket

HOST = '0.0.0.0'  # Listen on all laptop interfaces; client connects to actual laptop IP
PORT = 9999


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print('Waiting for EV3 on port {}...'.format(PORT))
        connection, address = server.accept()
        with connection:
            print('Connected:', address)
            with connection.makefile('r', encoding='utf-8') as responses:
                while True:
                    try:
                        entry = input('Target joint angles theta1 theta2 (or q): ').strip()
                    except (EOFError, KeyboardInterrupt):
                        entry = 'q'
                    if entry.lower() == 'q':
                        connection.sendall(b'EXIT\n')
                        break
                    try:
                        theta1, theta2 = map(float, entry.replace(',', ' ').split())
                    except ValueError:
                        print('Enter two angles, e.g. 30 -15')
                        continue
                    connection.sendall(('{:.4f},{:.4f}\n'.format(theta1, theta2)).encode('utf-8'))
                    reply = responses.readline()
                    if not reply:
                        print('EV3 disconnected')
                        break
                    print('Brick:', reply.strip())


if __name__ == '__main__':
    main()
