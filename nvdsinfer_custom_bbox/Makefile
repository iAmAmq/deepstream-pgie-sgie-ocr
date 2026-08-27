CXX := g++

NVDS_VERSION := 8.0
DS_ROOT := /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)

CXXFLAGS := -Wall -std=c++14 -shared -fPIC -O2
CXXFLAGS += -I$(DS_ROOT)/sources/includes -I/usr/local/cuda/include

LIBS := -L/usr/local/cuda/lib64 -lcudart

SRCFILES := nvdsparsebbox_truck.cpp
TARGET_LIB := libnvdsinfer_custom_bbox_truck.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

clean:
	rm -f $(TARGET_LIB)

.PHONY: all clean
