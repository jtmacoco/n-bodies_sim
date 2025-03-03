#include "particles.h"
void Particles::prepRender()
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * particles.size(), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}
void Particles::addParticle(float x, float y)
{
    std::shared_ptr<Particle> p(new Particle(x, y));
    particles.push_back(p);
}
void Particles::render()
{
    glBindVertexArray(VAO);
    std::vector<float> vertex_data;
    for (auto &p : particles)
    {
        p->update();
        float x = p->getPosition().x;
        float y = p->getPosition().y;
        vertex_data.push_back(x);
        vertex_data.push_back(y);
    }
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    /*
        note allocating space for the memory in prep can now can just fill the space with
        vertex data. SubData "reserve a specific amount of memory" CH.28 
    */
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.size() * sizeof(GLfloat), vertex_data.data());
    glPointSize(10.0f);
    glDrawArrays(GL_POINTS, 0, particles.size());
    glBindVertexArray(0);
}