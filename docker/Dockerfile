ARG BASE=railberkeley/robonetv2.base
FROM ${BASE}
ARG USER_ID UID GID

USER root
RUN usermod -u ${UID} ${USER_ID}
RUN groupmod -g ${GID} ${USER_ID}

USER ${USER_ID}
