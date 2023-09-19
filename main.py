

from src.app import create_demo


def main():
    demo = create_demo()

    demo.launch(server_name="127.0.0.1", server_port=8888)


if __name__ == "__main__":
    main()
