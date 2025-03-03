#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "particle.h"
#include "particles.h"
#define G 6.674e-11
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
float getDistance(float x1, float y1, float x2, float y2){
    float distance = std::sqrt(std::pow(x2-x1,2)+ std::pow(y2-y1,2));
    return distance;
}
float gravitationalForce(Particle p1, Particle p2){
    return 0.0;
}
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) // checks if a key has been pressed so this function should return the key pressed as a enum I think
        glfwSetWindowShouldClose(window, true);
}
std::string LoadShaderSource(const std::string &filepath)
{
    std::ifstream file(filepath);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
GLuint CompileShader(GLenum type, const std::string &source)
{
    GLuint shader = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Error: " << infoLog << std::endl;
    }
    return shader;
}
int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // for MAC
    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); // basically set up whiteboard we will be drawing on remeber analogy
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // will resize viewpoart when window re-sizes

    Particles ps;

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    
    std::stringstream buffer;
    /*
        Remember we need to first load the shaders for the vertex and fragment
        -The vertex shader processes the vertices, which is just a point(particle)
        -The fragment shader will handle the coloring of the rendered output
    */
    std::string vert_shader_content = LoadShaderSource("./shaders/simple.vert");
    std::string frag_shader_content = LoadShaderSource("./shaders/simple.frag");
    GLuint vert_shader = CompileShader(GL_VERTEX_SHADER, vert_shader_content);
    GLuint frag_shader = CompileShader(GL_FRAGMENT_SHADER, frag_shader_content);
    GLuint shaderProgram; // will link all shaders together
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vert_shader);
    glAttachShader(shaderProgram, frag_shader);
    glLinkProgram(shaderProgram);
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    }

    glUseProgram(shaderProgram);
    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    ps.addParticle(0.0f,0.0f);
    ps.prepRender();

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        processInput(window);
        glUseProgram(shaderProgram);
        ps.render();
        glfwSwapBuffers(window);
        glfwPollEvents(); // checks if events are triggered
    }
    glfwTerminate();
    return 0;
}