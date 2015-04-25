#ifndef GLSL_PROGRAM_H
#define GLSL_PROGRAM_H
#include <QOpenGLFunctions>
#include <glsl_shader.h>
#include <vector>
#include <glm/glm.hpp>
using std::vector;
using glm::vec2;
class GLSL_program
{
public:
    GLSL_program();
    ~GLSL_program();

    void add_shader(GLenum shader_type, const char *path);
    void load_vertex_shader(const char *path);
    void load_fragment_shader(const char *path);
    void use() const;
    void set_shader_attribute( const char *name, glm::vec2 value);
    void set_shader_attribute( const char *name, float value);
    void set_shader_attribute( const char *name, glm::mat4 & matrix);


private:
    GLuint m_program_handle;
    QOpenGLFunctions *f;

};

#endif // GLSL_PROGRAM_H
