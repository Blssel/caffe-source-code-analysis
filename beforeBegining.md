**转载于[All Posts – Xuesong's Blog](http://alanse7en.github.io/caffedai-ma-jie-xi-2/) 并重新排版和分析**



在Caffe中定义一个网络是通过编辑一个prototxt文件来完成的，一个简单的网络定义文件如下：

```
name: "ExampleNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "path/to/train_database"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "conv1"
  top: "ip1"
  inner_product_param {
    num_output: 500
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip1"
  bottom: "label"
  top: "loss"
}
```

这个网络定义了一个name为ExampleNet的网络，这个网络的输入数据是LMDB数据，batch_size为64，包含了一个卷积层和一个全连接层，训练的loss function为SoftmaxWithLoss。通过这种简单的key: value描述方式，用户可以很方便的定义自己的网络，利用Caffe来训练和测试网络，验证自己的想法。

Caffe中定义了丰富的layer类型，每个**类型都有对应的一些参数来描述这一个layer**。为了说明的方便，接下来将通过一个简单的例子来展示Caffe是如何使用**Google Protocol Buffer**来完成Solver和Net的定义。

首先我们需要了解Google Protocol Buffer定义data schema的方式，Google Protocol Buffer通过一种类似于C++的语言来定义数据结构，下面是官网上一个典型的AddressBook例子：

```
// AddressBook.proto
package tutorial;

message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    required string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phone = 4;
}

message AddressBook {
  repeated Person person = 1;
}
```

第2行的package tutorial类似于C++里的namespace，message可以简单的理解为一个class，message可以嵌套定义。每一个field除了一般的int32和string等类型外，还有一个属性来表明这个field是required,optional或repeated。required的field必须存在，相对应的optional的就可以不存在，**repeated的field可以出现0次或者多次**。

>  这一点对于Google Protocol Buffer的兼容性很重要，比如新版本的AddressBook添加了一个string类型的field，只有把这个field的属性设置为optional，就可以保证新版本的代码读取旧版本的数据也不会出错，新版本只会认为旧版本的数据没有提供这个optional field，会直接使用default。同时我们也可以定义enum类型的数据。

 **每个field等号右侧的数字可以理解为在实际的binary encoding中这个field对应的key值，通常的做法是将经常使用的field定义为0-15的数字**，可以节约存储空间（涉及到具体的encoding细节，感兴趣的同学可以看看官网的解释），其余的field使用较大的数值。



类似地在caffe/src/caffe/proto/中有一个caffe.proto文件，其中对layer的部分定义为：

```c++
message LayerParameter {
  optional string name = 1; // the layer name
  optional string type = 2; // the layer type
  repeated string bottom = 3; // the name of each bottom blob
  repeated string top = 4; // the name of each top blob
//  other fields
}
```

在定义好了data schema之后，需要使用`protoc compiler`来编译定义好的proto文件。常用的命令为：

```shell
protoc -I=/protofile/directory –cpp_out=/output/directory /path/to/protofile
```

`I`之后为`proto`文件的路径，`--cpp_out`为编译生成的`.h`和`.cc`文件的路径，最后是`proto`文件的路径。**编译之后会生成`AddressBook.pb.h`和`AddressBook/pb.cc`文件，其中包含了大量的接口函数**，用户可以利用这些接口函数获取和改变某个`field`的值。对应上面的`data schema`定义，有这样的一些接口函数：

```
// name
inline bool has_name() const;
inline void clear_name();
inline const ::std::string& name() const;  //getter
inline void set_name(const ::std::string& value);  //setter
inline void set_name(const char* value);  //setter
inline ::std::string* mutable_name();

// email
inline bool has_email() const;
inline void clear_email();
inline const ::std::string& email() const; //getter
inline void set_email(const ::std::string& value);  //setter
inline void set_email(const char* value);  //setter
inline ::std::string* mutable_email();

// phone
inline int phone_size() const;
inline void clear_phone();
inline const ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >& phone() const;
inline ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >* mutable_phone();
inline const ::tutorial::Person_PhoneNumber& phone(int index) const;
inline ::tutorial::Person_PhoneNumber* mutable_phone(int index);
inline ::tutorial::Person_PhoneNumber* add_phone();
```

每个类都有对应的`setter`和getter，因为phone是repeated类型的，所以还多了通过index来获取和改变某一个元素的setter和getter，phone还有一个获取数量的phone_size函数。官网上的tutorial是通过

`bool ParseFromIstream(istream* input);`来从`binary`的数据文件里解析数据，为了更好地说明`Caffe`中读取数据的方式，我稍微修改了代码，使用了和Caffe一样的方式通过`TextFormat::Parse`来解析文本格式的数据。具体的代码如下：

```C++
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "addressBook.pb.h"

using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

// Iterates through all people in the AddressBook and prints info about them.
void ListPeople(const tutorial::AddressBook& address_book) {
  for (int i = 0; i < address_book.person_size(); i++) {
    const tutorial::Person& person = address_book.person(i);

    cout << "Person ID: " << person.id() << endl;
    cout << "  Name: " << person.name() << endl;
    if (person.has_email()) {
      cout << "  E-mail address: " << person.email() << endl;
    }

    for (int j = 0; j < person.phone_size(); j++) {
      const tutorial::Person::PhoneNumber& phone_number = person.phone(j);

      switch (phone_number.type()) {
        case tutorial::Person::MOBILE:
          cout << "  Mobile phone #: ";
          break;
        case tutorial::Person::HOME:
          cout << "  Home phone #: ";
          break;
        case tutorial::Person::WORK:
          cout << "  Work phone #: ";
          break;
      }
      cout << phone_number.number() << endl;
    }
  }
}

// Main function:  Reads the entire address book from a file and prints all
//   the information inside.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    cerr << "Usage:  " << argv[0] << " ADDRESS_BOOK_FILE" << endl;
    return -1;
  }

  tutorial::AddressBook address_book;

  {
    // Read the existing address book.
    int fd = open(argv[1], O_RDONLY); //O_RDONLY:只读模式,O_WRONLY:只写模式,O_RDWR:读写模式
    FileInputStream* input = new FileInputStream(fd);
    if (!google::protobuf::TextFormat::Parse(input, &address_book)) {
      cerr << "Failed to parse address book." << endl;
      delete input;
      close(fd);
      return -1;
    }
  }

  ListPeople(address_book);

  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
```

读取和解析数据的代码：

```
int fd = open(argv[1], O_RDONLY);
FileInputStream* input = new FileInputStream(fd);
if (!google::protobuf::TextFormat::Parse(input, &address_book)) {
  cerr << "Failed to parse address book." << endl;
}
```

**这一段代码将input解析为我们设计的数据格式，写入到`address_book`**中。之后再调用`ListPeople`函数输出数据，来验证数据确实是按照我们设计的格式来存储和读取的。**`ListPeople`函数中使用了之前提到的各个`getter`**

接口函数。上面的文件的解析结果如图所示：

```
# ExampleAddressBook.prototxt
person {
  name: "Alex K"
  id: 1
  email: "kongming.liang@abc.com"
  phone {
    number: "+86xxxxxxxxxxx"
    type: MOBILE
  }
}

person {
  name: "Andrew D"
  id: 2
  email: "xuesong.deng@vipl.ict.ac.cn"
  phone {
    number: "+86xxxxxxxxxxx"
    type: MOBILE
  }
  phone {
    number: "+86xxxxxxxxxxx"
    type: WORK
  }
}
```

![img](http://alanse7en.github.io/images/listPerson.png)
