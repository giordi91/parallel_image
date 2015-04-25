#include <ui/glsl_program.h>

GLSL_program::GLSL_program():f(nullptr)
{
    f = QOpenGLContext::currentContext()->functions();
    m_program_handle = f->glCreateProgram();
    if( 0 == m_program_handle )
    {
        throw std::runtime_error("Error creating vertex shader.");
        return;
    }

    f->glLinkProgram( m_program_handle );


}

GLSL_program::~GLSL_program()
{
    f->glUseProgram(0);
    f->glDeleteProgram(m_program_handle);
}

void GLSL_program::add_shader(GLenum shader_type, const char *path)
{
    GLSL_shader shader(shader_type,path);
    f->glAttachShader(m_program_handle,
                      shader.get_shader_id());
    f->glLinkProgram( m_program_handle );
    GLint status;
    f->glGetProgramiv( m_program_handle, GL_LINK_STATUS, &status );

    //TODO throw exception?
    if( GL_FALSE == status )
    {
        fprintf( stderr, "Failed to link shader program!\n" );
        GLint logLen;
        f->glGetProgramiv(m_program_handle, GL_INFO_LOG_LENGTH, &logLen);
        if( logLen > 0 )
        {
            char * log = new char[logLen];
            GLsizei written;
            f->glGetProgramInfoLog(m_program_handle, logLen, &written, log);
            fprintf(stderr, "Program log: \n%s", log);
            delete [] log;
        }
    }

    f->glDeleteShader(shader.get_shader_id());

}

void GLSL_program::load_vertex_shader(const char *path)
{
    add_shader(GL_VERTEX_SHADER, path);
}

void GLSL_program::load_fragment_shader(const char *path)
{
    add_shader(GL_FRAGMENT_SHADER, path);
}

void GLSL_program::use() const
{
    f->glUseProgram(m_program_handle);
}


void GLSL_program::set_shader_attribute( const char *name, vec2 value)
{
    GLuint loc = f->glGetUniformLocation(m_program_handle, name);
    f->glUniform2f(loc,value.x,value.y);
}


void GLSL_program::set_shader_attribute( const char *name, float value)
{
    GLuint loc = f->glGetUniformLocation(m_program_handle, name);
    f->glUniform1f(loc,value);
}

void GLSL_program::set_shader_attribute( const char *name, glm::mat4 & matrix)
{
    GLuint loc = f->glGetUniformLocation(m_program_handle, name);
    f->glUniformMatrix4fv(loc, 1, GL_FALSE, &matrix[0][0]);
}

