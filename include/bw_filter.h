#include <tbb/blocked_range.h>


void bw_serial(	const uint8_t * source,
                uint8_t* target,
                const int &width,
                const int &height);


void bw_tbb(
			const uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height);


class Apply_bw_tbb
{
public:
	Apply_bw_tbb(const uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height);
	void operator() (const tbb::blocked_range<size_t>& r)const;

private:

	const uint8_t * m_source;
    uint8_t* m_target;
    const int m_width;
    const int m_height;
};