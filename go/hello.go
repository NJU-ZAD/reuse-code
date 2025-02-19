package main

import "fmt"

func main() {
	fmt.Println("hello")
	fmt.Println("go")
	fmt.Println("debug")
}

/*
cd go;go mod init hello;cd ..
cd go;go run hello.go;cd ..
cd go;go build hello.go;./hello;rm -rf hello;cd ..
*/
