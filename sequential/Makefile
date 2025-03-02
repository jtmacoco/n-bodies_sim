CXX = g++
CPPFLAGS = -g
CPPWARNINGS = -Wall -pedantic
DEPDIR = deps
OBJDIR = objs
LIBS = -lGLEW -lGL -lGLU -lglut -lglfw  
# Find all .cpp files and their respective directories
CPPSOURCEFILES = $(shell find . -name '*.cpp')
# Get the object files paths corresponding to source files
OBJFILES = $(CPPSOURCEFILES:%.cpp=$(OBJDIR)/%.o)
main: $(OBJFILES)
	$(CXX) -g -o main $(OBJFILES)  $(LIBS)

$(OBJDIR)/%.o : %.cpp  |  $(OBJDIR)
	$(CXX) $(CPPFLAGS) $(CPPWARNINGS) $(MACRO_DEFINITIONS) -o $@ -c $<
# Create the necessary directories
$(OBJDIR): ; @mkdir -p $@


# Include dependency files

# Clean up generated files
clean:
	rm -rf $(OBJDIR)  main