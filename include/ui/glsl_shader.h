#ifndef GLSL_SHADER_H
#define GLSL_SHADER_H
#include <QtGui/QOpenGLFunctions>
#include <string>

using std::string;

//TODO TRY PRE PROCESSOR MACRO FOR GETTER SETTER
/**
 * @brief The GLSL_shader class allows to manipulate shaders
 */
class GLSL_shader
{
public:
    /**
     * @brief The constructor
     * @param shader_type, what type of shader we are instantiating
     * @param the path to the shader
     * @thow : runtime_error if path not valid
     */
    GLSL_shader(GLenum shader_type, const char *path);

    ~GLSL_shader();
    /**
     * @brief getter function for the shader id
     * @return
     */
    GLuint get_shader_id() const;


protected:
    /**
     * @brief Function for loading shader source code
     * @param the path to the shader
     * @return std::string containing the source code
     */
    string load_shader_source(const char * path);

private:
    //the shader id
    GLuint m_shader_id;
    //the shader type
    GLenum m_shader_type;
};

#endif // GLSL_SHADER_H
