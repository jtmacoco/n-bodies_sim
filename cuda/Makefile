CXX = nvcc
CXX_CPP = g++  # Compiler for C++ files

DEPDIR = deps
OBJDIR = objs
LIBS = -lGLEW -lGL -lGLU -lglut -lglfw -lcudadevrt 

CU_SOURCES = $(shell find . -name '*.cu')
CPP_SOURCES = $(shell find . -name '*.cpp')

CU_OBJECTS = $(CU_SOURCES:%.cu=$(OBJDIR)/%.o)
CPP_OBJECTS = $(CPP_SOURCES:%.cpp=$(OBJDIR)/%.o)
OBJFILES = $(CU_OBJECTS) $(CPP_OBJECTS)

main: $(OBJFILES)
	$(CXX) -g -o main $(OBJFILES) $(LIBS)

$(OBJDIR)/%.o : %.cu | $(DEPDIR) $(OBJDIR)
	$(CXX)  --relocatable-device-code true -c $< -o $@

$(OBJDIR)/%.o : %.cpp | $(DEPDIR) $(OBJDIR)
	$(CXX_CPP) -c $< -o $@

# Create necessary directories
$(DEPDIR): ; @mkdir -p $@
$(OBJDIR): ; @mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(DEPDIR) main
