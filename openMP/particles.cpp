#include "particles.h"
void Particles::prepRender()
{
    initSystem();
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * particles.size(), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void Particles::render(float dt)
{
    dt = 0.0001; // maybe adjust
    glBindVertexArray(VAO);
    std::vector<float> vertex_data;
    sumForces(dt);
    for (auto &p : particles)
    {
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
    glPointSize(particle_size);
    glDrawArrays(GL_POINTS, 0, particles.size());
    glBindVertexArray(0);
}
void Particles::addParticle(float mass)
{
    glm::vec3 pos = randCircle();
    glm::vec3 vel = randCircle();
    std::shared_ptr<Particle> p(new Particle(pos, vel, mass)); // change this to glm::vec3 for pos and vel
    particles.push_back(p);
}


glm::vec3 Particles::randCircle()
{
    static std::uniform_real_distribution<float> dist_theta(0.0f, 2.0f * M_PI);
    static std::uniform_real_distribution<float> dist_radius(0.0f, 0.8f);

    float theta = dist_theta(gen);
    float r = dist_radius(gen);
    return glm::vec3(std::cos(theta), std::sin(theta), 0.0f) * r;
}
/*summing up the forces*/
void Particles::sumForces(float dt)
{
    std::vector<glm::vec3> force_sums(particles.size(), glm::vec3(0.0f));

    #pragma omp parallel for
    for (int i = 0; i < int(particles.size()); i++)
    {
        std::shared_ptr<Particle> cur_particle = particles[i];
        glm::vec3 force_sum(0.0f);
        for (int j = i + 1; j < int(particles.size()); j++)
        {
            force_sum += gravitationalForce(*cur_particle, *particles[j]);
            // printf("x: %f, y: %f \n",force_sum.x,force_sum.y);
        }
        force_sums[i]=force_sum;
        //cur_particle->applyForce(force_sum, dt);
    }
    #pragma omp parallel for
    for(int i = 0; i <int(particles.size()); i++){
         std::shared_ptr<Particle> cur_particle = particles[i];
         cur_particle->applyForce(force_sums[i],dt);
    }

}
glm::vec3 Particles::gravitationalForce(Particle p1, Particle p2)
{
    float eps = 0.1f; // needed so force doesn't get to big shooting particles everywhere
    glm::vec3 p1_pos = p1.getPosition();
    float p1_mass = p1.getMass();

    glm::vec3 p2_pos = p2.getPosition();
    float p2_mass = p2.getMass();

    glm::vec3 distance = p2_pos - p1_pos; // distance between 2 particles

    float r = glm::length(distance); //  length of the vector

    glm::vec3 r_unit = glm::normalize(distance);                       // unit vector
    float force = G * ((p1_mass * p2_mass) / ((r * r) + (eps * eps))); // gravitational force equation
    return force * r_unit;                                             // multiple by unit vector for direction of the force so provides direction
}
/*grab average velocity over all particles in the system*/
void Particles::initVel()
{
    avg_vel = glm::vec3(0.0f);

    #pragma omp parallel for
    for (auto &p : particles)
    {
        avg_vel += p->getVelocity() * p->getMass();
    }
    avg_vel /= particles.size();
}
/*grab center of mass of the system basically*/
void Particles::initPos()
{
    center_mass = glm::vec3(0.0f);
    #pragma omp parallel for
    for (auto &p : particles)
    {
        center_mass += p->getPosition() * p->getMass();
    }
    center_mass /= particles.size();
}
/*This is meant to intilize the conditions at the start of the simulation.*/
void Particles::initSystem()
{
    initVel();
    initPos();
    #pragma omp parallel for
    for (auto &p : particles)
    {
        p->setPosition((center_mass)); // note not realy setting but doing p pos-= center of mass
        p->setVelocity(avg_vel);       // note not really setting but doing  p vel -= avg_vel
    }
}